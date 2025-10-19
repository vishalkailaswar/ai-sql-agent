import os
import sqlite3
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, inspect
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
import tempfile

# -----------------------------
# 1️⃣ PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="AI SQL Agent Explorer",
    page_icon="🤖",
    layout="wide"
)
st.title("🤖 AI SQL Agent Explorer")
st.markdown("Upload a **CSV**, **Parquet**, or **SQLite DB** file, explore it, and ask natural language questions!")

# -----------------------------
# 2️⃣ GEMINI API CONFIG
# -----------------------------
# os.environ["GEMINI_API_KEY"] = "AIzaSyA-YLDbcryJkmZJN2CAAf-6gEhQYhnCgPw"  # 🔑 Replace with your real key
import streamlit as st
import os
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]


# -----------------------------
# 3️⃣ FILE UPLOAD SECTION
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a data file (CSV, Parquet, or .db):",
    type=["csv", "parquet", "db", "sqlite"]
)

if uploaded_file is not None:
    # Create a temp file for database handling
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "temp_data.db")

    try:
        file_name = uploaded_file.name.lower()

        # --- CASE 1: CSV ---
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            table_name = "uploaded_csv"
            engine = create_engine(f"sqlite:///{db_path}")
            df.to_sql(table_name, con=engine, index=False, if_exists="replace")
            st.success(f"✅ CSV file uploaded and stored as `{table_name}` table in temporary SQLite DB.")

        # --- CASE 2: Parquet ---
        elif file_name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)
            table_name = "uploaded_parquet"
            engine = create_engine(f"sqlite:///{db_path}")
            df.to_sql(table_name, con=engine, index=False, if_exists="replace")
            st.success(f"✅ Parquet file uploaded and stored as `{table_name}` table in temporary SQLite DB.")

        # --- CASE 3: Existing SQLite DB ---
        elif file_name.endswith((".db", ".sqlite")):
            db_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(db_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            engine = create_engine(f"sqlite:///{db_path}")
            st.success(f"✅ SQLite DB file uploaded and ready for use.")

        else:
            st.error("Unsupported file type. Please upload CSV, Parquet, or SQLite DB.")
            st.stop()

        # -----------------------------
        # 4️⃣ INSPECT DATABASE
        # -----------------------------
        inspector = inspect(engine)
        db = SQLDatabase(engine)

        st.subheader("📊 Database Overview")

        tables = inspector.get_table_names()
        st.write(f"**Detected Tables:** {len(tables)}")

        for table in tables:
            columns = inspector.get_columns(table)
            col_details = pd.DataFrame(columns)[["name", "type"]]
            with st.expander(f"📁 {table}"):
                st.dataframe(col_details, use_container_width=True)

        # -----------------------------
        # 5️⃣ GEMINI + LANGCHAIN AGENT
        # -----------------------------
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-preview-09-2025",
                temperature=0.2,
                google_api_key=os.environ["GEMINI_API_KEY"]
            )

            agent_executor = create_sql_agent(
                llm=llm,
                db=db,
                agent_type="zero-shot-react-description",
                verbose=True,
                agent_executor_kwargs={"handle_parsing_errors": True}
            )

            st.success("🤖 Gemini SQL Agent initialized successfully!")

        except Exception as e:
            st.error(f"❌ Error initializing Gemini Agent: {e}")
            st.stop()

        # -----------------------------
        # 6️⃣ USER QUERY INPUT
        # -----------------------------
        st.markdown("---")
        st.subheader("💬 Ask a Question")
        query_input = st.text_area(
            "Type your question below:",
            placeholder="e.g., What is the average price by category?",
            height=100
        )

        if st.button("🚀 Generate Answer"):
            if not query_input.strip():
                st.warning("Please enter a question first.")
            else:
                with st.spinner("Analyzing your question..."):
                    try:
                        response = agent_executor.invoke({"input": query_input})
                        final_answer = response["output"]

                        st.success("✅ Answer generated successfully!")
                        st.markdown("### 🧠 AI Response:")
                        st.write(final_answer)

                    except Exception as e:
                        st.error(f"❌ Error during execution: {e}")

        # -----------------------------
        # 7️⃣ SAMPLE DATA PREVIEW
        # -----------------------------
        st.markdown("---")
        st.subheader("🗂️ Preview Sample Data")
        selected_table = st.selectbox("Select a table to preview:", tables)
        if selected_table:
            df_preview = pd.read_sql_query(f"SELECT * FROM {selected_table} LIMIT 10", engine)
            st.dataframe(df_preview, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

else:
    st.info("👆 Upload a file to get started.")
