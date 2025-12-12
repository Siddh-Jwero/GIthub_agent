#agent.py
import zipfile
import os
import hashlib
import requests
from pathlib import Path
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import CustomLLM, LLMMetadata, CompletionResponse
import logging
from typing import List, Any, Generator, AsyncGenerator

# --------------------------------------------------------------
# 1. SILENCE NOISY LOGS
# --------------------------------------------------------------
logging.getLogger("llama_index").setLevel(logging.CRITICAL)

# --------------------------------------------------------------
# 2. IMPORT FAST EMBEDDER (ModernBERT / MLX)
# --------------------------------------------------------------
from embedder import embedder  # <-- ModernBERT 4-bit local embedder

# --------------------------------------------------------------
# 3. LLAMA-INDEX EMBEDDING WRAPPER
# --------------------------------------------------------------
from llama_index.core.embeddings import BaseEmbedding

class LlamaIndexWrapper(BaseEmbedding):
    def __init__(self, dim: int = 768):
        super().__init__()
        self._dimension = dim

    def _get_query_embedding(self, query: str) -> List[float]:
        return embedder.embed_query(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return embedder.embed_query(text)

    def _get_text_embedding_batch(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        return embedder.embed_documents(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embedding_batch(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        return self._get_text_embedding_batch(texts, **kwargs)

    @property
    def dimension(self) -> int:
        return self._dimension


embed_model = LlamaIndexWrapper(dim=768)

# --------------------------------------------------------------
# 4. CONFIG
# --------------------------------------------------------------
TEMP_DIR = "temp_repo"
OUTPUT_DIR = "output"
LLM_API = os.getenv("OPENAI_API_BASE")

# --------------------------------------------------------------
# 5. HELPER — convert any Response to string
# --------------------------------------------------------------
def to_text(resp):
    """Convert LlamaIndex Response or string-like objects safely to text."""
    if resp is None:
        return ""
    if hasattr(resp, "response"):
        return resp.response
    if hasattr(resp, "text"):
        return resp.text
    return str(resp)

# --------------------------------------------------------------
# 6. AUTO-DETECT MODEL FROM LM-STUDIO
# --------------------------------------------------------------
def get_lmstudio_model():
    try:
        r = requests.get(f"{LLM_API}/models", timeout=5)
        if r.status_code == 200:
            models = r.json().get("data", [])
            if models:
                return models[0]["id"]
    except Exception as e:
        st.warning(f"Auto-detect failed: {e}. Using default model.")
    return "openai/gpt-5.1"

# --------------------------------------------------------------
# 7. CUSTOM LLM (LM-STUDIO API)
# --------------------------------------------------------------
class LMStudioLLM(CustomLLM):
    model_name: str
    temperature: float = 0.7
    context_window: int = 32768
    num_output: int = -1
    model_config = {"extra": "allow"}

    def __init__(self, model_name: str, temperature: float = 0.7):
        super().__init__(model_name=model_name, temperature=temperature)
        self.base_url = os.getenv("OPENAI_API_BASE")
        self.api_key = os.getenv("OPENAI_API_KEY")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=self.context_window, num_output=self.num_output)

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.num_output,
            "stream": False,
            **kwargs,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions", 
            json=payload, 
            headers=headers,
            timeout=300
        )
        resp.raise_for_status()
        text = resp.json()["choices"]["message"]["content"]
        return CompletionResponse(text=text)

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return self.complete(prompt, **kwargs)

    def stream_complete(self, prompt: str, **kwargs: Any) -> Generator[CompletionResponse, None, None]:
        yield self.complete(prompt, **kwargs)

    async def astream_complete(self, prompt: str, **kwargs: Any) -> AsyncGenerator[CompletionResponse, None]:
        yield self.complete(prompt, **kwargs)


# --------------------------------------------------------------
# 8. EXTRACT ZIP (clean old files first)
# --------------------------------------------------------------
def extract_repo(zip_path: str):
    os.makedirs(TEMP_DIR, exist_ok=True)
    import shutil
    for item in os.listdir(TEMP_DIR):
        p = os.path.join(TEMP_DIR, item)
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.unlink(p)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(TEMP_DIR)
    st.success(f"✅ Extracted → `{TEMP_DIR}`")

# --------------------------------------------------------------
# 9. BUILD INDEX (NO CACHING)
# --------------------------------------------------------------
def build_index(_repo_hash: str):
    import shutil
    if not os.path.isdir(TEMP_DIR) or not os.listdir(TEMP_DIR):
        st.error("No files extracted!")
        return None

    # Clear any old persisted index
    storage_dir = "storage"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)

    docs = SimpleDirectoryReader(
        TEMP_DIR,
        recursive=True,
        exclude=[
        "*.test.py", "*__pycache__*", "*.pyc", "*.log",
        "*.mp3", "*.wav", "*.m4a", "*.mp4", "*.mov", "*.avi", "*.flac", "*.ogg",
        "node_modules", ".git", ".venv", "*.md"
        ],
    ).load_data()

    if not docs:
        st.error("No documents loaded!")
        return None

    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(docs)

    index = VectorStoreIndex(nodes, embed_model=embed_model, embed_batch_size=32)
    st.success("✅ Index built into Embeddings")
    return index

# --------------------------------------------------------------
# 10. MAIN STREAMLIT APP
# --------------------------------------------------------------
def main():
    st.title("🤖 AI Codebase → Docs Agent")

    auto_detected = get_lmstudio_model()
    st.info(f"**Auto-detected model:** `{auto_detected}`")

    available_models = [
        auto_detected,
    ]
    selected_model = st.selectbox("Select LLM (loaded in LM-Studio)", available_models, index=0)
    llm = LMStudioLLM(model_name=selected_model, temperature=0.7)

    uploaded = st.file_uploader("📦 Upload GitHub Repo (.zip)", type="zip")
    if not uploaded:
        st.info("Upload a .zip → Click **Start Analysis**")
        return

    zip_path = "repo.zip"
    with open(zip_path, "wb") as f:
        f.write(uploaded.getbuffer())

    if st.button("🚀 Start Analysis"):
        with st.spinner("Extracting repository..."):
            extract_repo(zip_path)

        repo_hash = hashlib.md5(open(zip_path, "rb").read()).hexdigest()

        with st.spinner("Building knowledge base..."):
            index = build_index(repo_hash)
            if not index:
                return
            engine = index.as_query_engine(llm=llm)

        # === 1. Overview ===
        with st.expander("📘 1. Project Overview", expanded=True):
            overview = to_text(engine.query(
                "Analyze the codebase and summarize:\n"
                "- Project name\n- Description\n- Tech stack\n- Entry point\n- Folder structure overview."
            ))
            st.markdown(overview)
            st.session_state.overview = overview

        # === 2. Generate README ===
        with st.expander("🧾 2. Generate README.md", expanded=True):
            readme = to_text(engine.query(
                f"Using this project overview:\n{st.session_state.overview}\n\n"
                "Generate a **professional and structured README.md** including:\n"
                "- # Title\n- ## Description\n- ## Features\n- ## Installation\n"
                "- ## Usage\n- ## API Reference\n- ## Folder Structure (in ##Folder Structure section/block)\n- ## Contributing\n"
                "- ## License\nEnsure Markdown syntax is perfect with spacing and headers."
            ))
            st.markdown(readme)
            st.session_state.readme = readme

        # === 3. Verification & Auto-Fix ===
        with st.expander("🔍 3. Self-Verification & Auto-Fix", expanded=True):
            check = to_text(engine.query(
                f"README:\n{st.session_state.readme}\n\n"
                "Review ALL code files and verify the README accuracy.\n"
                "Check for incorrect function/class names, wrong dependencies, or invalid setup steps.\n"
                "If issues are found, summarize them clearly. Otherwise say 'ALL CORRECT'."
            ))
            st.markdown(check)

            if "all correct" not in check.lower():
                st.warning("Fixing README automatically...")
                fixed = to_text(engine.query(
                    f"Fix and improve the README.md based on these verification results:\n{check}\n\n"
                    f"Here is the original README:\n{st.session_state.readme}\n\n"
                    "Ensure the final version is perfectly formatted Markdown, with consistent headings and spacing."
                ))
                st.success("✅ Fixed README generated!")
                st.markdown("**Final README.md:**")
                st.markdown(fixed)
                st.session_state.readme_fixed = fixed
            else:
                st.session_state.readme_fixed = st.session_state.readme
                st.success("✅ README verified as correct!")

        # === 4. Architecture Diagram ===
        with st.expander("🧩 4. Architecture Diagram", expanded=True):
            diag = to_text(engine.query(
                "Generate a **Mermaid** flowchart of the application architecture:\n"
                "- Components and relationships\n- Data flow\n- APIs / Services / DB\n"
                "Return only valid Markdown with ```mermaid code block."
            ))
            st.code(diag, language="mermaid")





        # === 5. Export ===
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        readme_original_path = Path(f"{OUTPUT_DIR}/README_original.md")
        readme_fixed_path = Path(f"{OUTPUT_DIR}/README_final.md")
        diagram_path = Path(f"{OUTPUT_DIR}/ARCHITECTURE.mmd")

        readme_original_path.write_text(st.session_state.readme)
        readme_fixed_path.write_text(st.session_state.readme_fixed)
        diagram_path.write_text(diag)
        
        st.success(f"📁 Exported all files → `{OUTPUT_DIR}/`")

        # --- 🪄 Download Buttons ---
        st.markdown("### 📥 Download Your Files")

        with open(readme_fixed_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Final README.md",
                data=f,
                file_name="README.md",
                mime="text/markdown",
            )

        with open(readme_original_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Original README.md",
                data=f,
                file_name="README_original.md",
                mime="text/markdown",
            )

        with open(diagram_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Architecture Diagram (.mmd)",
                data=f,
                file_name="ARCHITECTURE.mmd",
                mime="text/plain",
            )

        st.info("✅ You can also find these files saved in the `output/` folder locally.")


# --------------------------------------------------------------
if __name__ == "__main__":
    main()
