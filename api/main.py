import os
import re
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

from google import genai
from google.genai import types
from groq import Groq


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "BioRAG")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "documents")

VECTOR_INDEX = os.getenv(
    "VECTOR_INDEX",
    "vector_index_filtered"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

TOP_K = int(os.getenv("TOP_K", "10"))

RETRIEVAL_POOL = int(
    os.getenv(
        "RETRIEVAL_POOL",
        "40"
    )
)

MAX_CONTEXT_CHARS = int(
    os.getenv(
        "MAX_CONTEXT_CHARS",
        "10000"
    )
)

if not MONGO_URI:
    raise RuntimeError(
        "MONGO_URI is missing from .env"
    )

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing from .env"
    )


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="BioRAG API",
    version="FINAL-1.0",
    description="Biomedical Drug Repurposing Intelligence API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# REQUEST
# ============================================================

class RepurposingRequest(BaseModel):

    drug: str

    question: str

    top_k: Optional[int] = 10


# ============================================================
# GLOBALS
# ============================================================

mongo_client = None
collection = None
embedding_model = None
gemini_client = None
groq_client = None


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup():

    global mongo_client
    global collection
    global embedding_model
    global gemini_client
    global groq_client

    print()
    print("=" * 70)
    print("BIORAG API")
    print("=" * 70)

    print("Connecting to MongoDB...")

    mongo_client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=10000
    )

    mongo_client.admin.command("ping")

    collection = mongo_client[
        MONGO_DB
    ][
        MONGO_COLLECTION
    ]

    print(
        f"Connected to "
        f"{MONGO_DB}.{MONGO_COLLECTION}"
    )

    print("Loading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("Embedding model loaded.")

    print(
        f"Initializing Gemini: "
        f"{GEMINI_MODEL}"
    )

    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    print("Gemini ready.")

    if GROQ_API_KEY:

        groq_client = Groq(
            api_key=GROQ_API_KEY
        )

        print(
            f"Groq fallback ready: "
            f"{GROQ_MODEL}"
        )

    else:

        print(
            "Groq fallback disabled."
        )

    print()
    print("=" * 70)
    print("BIORAG READY")
    print("=" * 70)
    print()


# ============================================================
# SHUTDOWN
# ============================================================

@app.on_event("shutdown")
def shutdown():

    global mongo_client

    if mongo_client:

        mongo_client.close()


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize(value):

    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value)
        .strip()
        .lower()
    )


def format_value(value):

    if value is None:
        return None

    if isinstance(value, list):

        output = []

        for item in value:

            if isinstance(
                item,
                dict
            ):

                if item.get("disease"):
                    output.append(
                        str(
                            item["disease"]
                        )
                    )

                elif item.get("name"):
                    output.append(
                        str(
                            item["name"]
                        )
                    )

                else:
                    output.append(
                        str(item)
                    )

            else:

                output.append(
                    str(item)
                )

        return ", ".join(output)

    if isinstance(value, dict):

        if value.get("disease"):
            return str(
                value["disease"]
            )

        if value.get("name"):
            return str(
                value["name"]
            )

        return str(value)

    return str(value).strip()


# ============================================================
# METADATA
# ============================================================

def get_metadata(
    document,
    field,
    default=None
):

    # Top level
    value = document.get(
        field
    )

    if value is not None:
        return value

    # metadata
    metadata = document.get(
        "metadata"
    )

    if isinstance(
        metadata,
        dict
    ):

        value = metadata.get(
            field
        )

        if value is not None:
            return value

    # data
    data = document.get(
        "data"
    )

    if isinstance(
        data,
        dict
    ):

        value = data.get(
            field
        )

        if value is not None:
            return value

    return default


# ============================================================
# DRUG ALIASES
# ============================================================

DRUG_ALIASES = {

    "capmatinib": {
        "capmatinib",
        "inc280"
    },

    "imatinib": {
        "imatinib",
        "gleevec",
        "glivec",
        "sti571",
        "sti-571"
    }

}


# ============================================================
# STRICT DRUG MATCH
# ============================================================

def actual_document_drug(
    document
):

    value = get_metadata(
        document,
        "drug"
    )

    if value:

        return normalize(
            format_value(value)
        )

    # Some documents may use intervention_drug
    value = get_metadata(
        document,
        "intervention_drug"
    )

    if value:

        return normalize(
            format_value(value)
        )

    return ""


def document_is_about_drug(
    document,
    requested_drug
):

    requested = normalize(
        requested_drug
    )

    actual = actual_document_drug(
        document
    )

    aliases = DRUG_ALIASES.get(
        requested,
        {requested}
    )

    # --------------------------------------------------------
    # STRICT RULE:
    # If the document explicitly identifies another drug,
    # reject it.
    # --------------------------------------------------------

    if actual:

        actual_tokens = set(
            re.findall(
                r"[a-z0-9-]+",
                actual
            )
        )

        alias_tokens = set()

        for alias in aliases:

            alias_tokens.update(
                re.findall(
                    r"[a-z0-9-]+",
                    alias
                )
            )

        # Direct / alias match
        if (
            actual in aliases
            or
            bool(
                actual_tokens
                &
                alias_tokens
            )
        ):

            return True

        # It is explicitly another drug.
        return False

    # --------------------------------------------------------
    # No explicit drug field.
    #
    # Only then inspect document text.
    # --------------------------------------------------------

    text_parts = [

        get_metadata(
            document,
            "title",
            ""
        ),

        get_metadata(
            document,
            "text",
            ""
        ),

        get_metadata(
            document,
            "description",
            ""
        )

    ]

    searchable = normalize(
        " ".join(
            format_value(x) or ""
            for x in text_parts
        )
    )

    # Match aliases only if there is no explicit
    # drug field.
    for alias in aliases:

        if normalize(alias) in searchable:

            return True

    return False


# ============================================================
# ESTABLISHED INDICATION
# ============================================================

def extract_indication(
    text
):

    if not text:
        return None

    text = str(text)

    patterns = [

        r"Known\s+indications?\s*:"
        r"\s*(.*?)(?:\n\n|\Z)",

        r"Indications?\s*:"
        r"\s*(.*?)(?:\n\n|\Z)",

        r"indicated\s+for\s+the\s+treatment\s+of\s+"
        r"(.*?)(?:\.|\n)"

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE |
            re.DOTALL
        )

        if match:

            value = (
                match.group(1)
                .strip()
            )

            if value:
                return value

    return None


def get_established_indication(
    drug
):

    drug = normalize(drug)

    document = collection.find_one({

        "drug":
            drug,

        "type":
            "drug_profile"

    })

    if not document:

        document = collection.find_one({

            "drug":
                drug

        })

    if not document:
        return None

    for field in [

        "known_indications",
        "indication",
        "indications",
        "disease"

    ]:

        value = format_value(
            get_metadata(
                document,
                field
            )
        )

        if value:
            return value

    for field in [

        "text",
        "description",
        "content"

    ]:

        text = get_metadata(
            document,
            field
        )

        result = extract_indication(
            text
        )

        if result:
            return result

    return None


# ============================================================
# SEARCH
# ============================================================

def retrieve_documents(

    drug,
    question

):

    search_text = f"""
Drug: {drug}

Question:
{question}

Find clinical and biomedical evidence concerning
{drug} in diseases other than its established indication.

Prioritize:
clinical trials, published literature, efficacy,
treatment response, disease-specific investigations,
Phase 1, Phase 2, Phase 3, and combination studies.
""".strip()

    vector = embedding_model.encode(

        search_text,

        normalize_embeddings=True

    ).tolist()

    pipeline = [

        {

            "$vectorSearch": {

                "index":
                    VECTOR_INDEX,

                "path":
                    "embedding",

                "queryVector":
                    vector,

                "numCandidates":
                    max(
                        RETRIEVAL_POOL * 10,
                        200
                    ),

                "limit":
                    RETRIEVAL_POOL

            }

        },

        {

            "$addFields": {

                "vector_score": {

                    "$meta":
                        "vectorSearchScore"

                }

            }

        },

        {

            "$project": {

                "_id": 0,

                "document_id": 1,
                "drug": 1,
                "type": 1,
                "disease": 1,
                "conditions": 1,
                "title": 1,
                "phase": 1,
                "status": 1,
                "trial_id": 1,
                "nct_id": 1,
                "text": 1,
                "description": 1,
                "metadata": 1,
                "data": 1,
                "vector_score": 1

            }

        }

    ]

    return list(
        collection.aggregate(
            pipeline
        )
    )


# ============================================================
# ESTABLISHED INDICATION FILTER
# ============================================================

def is_established(
    document,
    indication
):

    if not indication:
        return False

    indication = normalize(
        indication
    )

    disease = normalize(
        format_value(
            get_metadata(
                document,
                "disease",
                ""
            )
        )
    )

    conditions = normalize(
        format_value(
            get_metadata(
                document,
                "conditions",
                ""
            )
        )
    )

    title = normalize(
        format_value(
            get_metadata(
                document,
                "title",
                ""
            )
        )
    )

    combined = (
        disease
        + " "
        + conditions
        + " "
        + title
    )

    # Explicit NSCLC handling
    if (
        "non-small cell lung"
        in indication
        and
        "non-small cell lung"
        in combined
    ):

        return True

    # General overlap
    indication_words = set(
        re.findall(
            r"[a-z]+",
            indication
        )
    )

    document_words = set(
        re.findall(
            r"[a-z]+",
            combined
        )
    )

    stopwords = {

        "the",
        "and",
        "of",
        "for",
        "with",
        "disease",
        "diseases",
        "cancer",
        "cancers",
        "chronic",
        "acute"

    }

    indication_words -= stopwords

    if not indication_words:
        return False

    overlap = (

        len(
            indication_words
            &
            document_words
        )
        /
        len(
            indication_words
        )

    )

    return overlap >= 0.65


# ============================================================
# EVIDENCE LEVEL
# ============================================================

def evidence_level(
    document
):

    phase = normalize(
        format_value(
            get_metadata(
                document,
                "phase",
                ""
            )
        )
    )

    status = normalize(
        format_value(
            get_metadata(
                document,
                "status",
                ""
            )
        )
    )

    doc_type = normalize(
        format_value(
            get_metadata(
                document,
                "type",
                ""
            )
        )
    )

    if (
        "withdrawn" in status
        or
        "terminated" in status
    ):

        return "LOW"

    if "phase 3" in phase:
        return "HIGH"

    if "phase 2" in phase:
        return "MEDIUM"

    if "phase 1" in phase:
        return "LOW"

    if "literature" in doc_type:
        return "MEDIUM"

    return "LOW"


# ============================================================
# DOCUMENT TEXT
# ============================================================

def document_text(
    document
):

    parts = []

    for field in [

        "type",
        "disease",
        "conditions",
        "title",
        "phase",
        "status",
        "trial_id",
        "nct_id"

    ]:

        value = format_value(
            get_metadata(
                document,
                field
            )
        )

        if value:

            parts.append(
                f"{field}: {value}"
            )

    text = format_value(
        get_metadata(
            document,
            "text",
            ""
        )
    )

    if text:

        parts.append(
            "evidence: "
            + text
        )

    return "\n".join(
        parts
    )


# ============================================================
# CONTEXT
# ============================================================

def build_context(
    documents
):

    blocks = []

    size = 0

    for i, document in enumerate(
        documents,
        1
    ):

        block = f"""
EVIDENCE {i}

Document ID:
{document.get("document_id", "N/A")}

Drug:
{format_value(document.get("drug", "N/A"))}

Type:
{format_value(get_metadata(document, "type", "N/A"))}

Disease:
{format_value(get_metadata(document, "disease", "N/A"))}

Conditions:
{format_value(get_metadata(document, "conditions", "N/A"))}

Title:
{format_value(get_metadata(document, "title", "N/A"))}

Phase:
{format_value(get_metadata(document, "phase", "N/A"))}

Status:
{format_value(get_metadata(document, "status", "N/A"))}

NCT:
{format_value(get_metadata(document, "nct_id", "N/A"))}

Vector score:
{document.get("vector_score", 0)}

Evidence level:
{evidence_level(document)}

Evidence:
{document_text(document)}
""".strip()

        if (
            size + len(block)
            >
            MAX_CONTEXT_CHARS
        ):
            break

        blocks.append(
            block
        )

        size += len(block)

    return "\n\n".join(
        blocks
    )


# ============================================================
# LLM PROMPT
# ============================================================

def build_prompt(

    drug,
    question,
    indication,
    context

):

    return f"""
You are BioRAG, a biomedical drug-repurposing evidence synthesis system.

Analyze ONLY the retrieved evidence.

Be scientifically conservative.

IMPORTANT:

- A clinical trial does NOT prove efficacy.
- A trial means the drug was investigated.
- Do not invent outcomes.
- Do not invent patient numbers.
- Do not invent response rates.
- Do not claim approval.
- Terminated or withdrawn trials are not successful evidence.
- Combination studies cannot automatically attribute efficacy
  to the requested drug.
- PK or safety studies are not repurposing efficacy evidence.
- Clearly distinguish investigation from demonstrated efficacy.

Drug:
{drug}

Established indication:
{indication}

Question:
{question}

Retrieved evidence:
{context}

Return:

# Repurposing Evidence for {drug}

## Summary

## Established Indication

## Potential Repurposing Signals

For each meaningful candidate:

### Disease

- Evidence
- Trial / paper
- Phase
- Status
- What was studied
- What the evidence supports
- Evidence strength
- Limitations

## Ranking of Signals

## Evidence Gaps

## Conclusion
""".strip()


# ============================================================
# GEMINI
# ============================================================

def generate_gemini(
    prompt
):

    response = (

        gemini_client
        .models
        .generate_content(

            model=GEMINI_MODEL,

            contents=prompt,

            config=types.GenerateContentConfig(

                temperature=0.2,

                max_output_tokens=2500

            )

        )

    )

    if not response.text:

        raise RuntimeError(
            "Gemini returned empty output."
        )

    return response.text


# ============================================================
# GROQ
# ============================================================

def generate_groq(
    prompt
):

    if not groq_client:

        raise RuntimeError(
            "Groq is unavailable."
        )

    response = (

        groq_client
        .chat
        .completions
        .create(

            model=GROQ_MODEL,

            messages=[

                {

                    "role":
                        "system",

                    "content":
                        (
                            "You are a "
                            "biomedical evidence "
                            "synthesis assistant. "
                            "Never invent evidence."
                        )

                },

                {

                    "role":
                        "user",

                    "content":
                        prompt

                }

            ],

            temperature=0.2,

            max_tokens=2500

        )

    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# ============================================================
# LLM ROUTER
# ============================================================

def generate_answer(
    prompt
):

    # Gemini
    try:

        print(
            f"Generating answer with "
            f"{GEMINI_MODEL}..."
        )

        answer = generate_gemini(
            prompt
        )

        print(
            "Primary model used: "
            f"{GEMINI_MODEL}"
        )

        return answer

    except Exception as e:

        print(
            "Gemini failed:"
        )

        print(
            str(e)
        )

    # Groq
    try:

        print(
            f"Falling back to "
            f"{GROQ_MODEL}..."
        )

        answer = generate_groq(
            prompt
        )

        print(
            "Fallback model used: "
            f"{GROQ_MODEL}"
        )

        return answer

    except Exception as e:

        print(
            "Groq failed:"
        )

        print(
            str(e)
        )

        raise RuntimeError(
            "Both LLM providers failed."
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "name":
            "BioRAG",

        "status":
            "running",

        "version":
            "FINAL-1.0"

    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "mongodb":
            collection is not None,

        "embedding_model":
            embedding_model is not None,

        "gemini":
            gemini_client is not None,

        "groq_fallback":
            groq_client is not None

    }


# ============================================================
# REPURPOSING API
# ============================================================

@app.post(
    "/api/repurposing"
)
def repurposing(
    request: RepurposingRequest
):

    drug = request.drug.strip()

    question = request.question.strip()

    if not drug:

        raise HTTPException(
            status_code=400,
            detail="Drug is required."
        )

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question is required."
        )

    top_k = min(
        max(
            request.top_k or 10,
            5
        ),
        20
    )

    print()
    print("=" * 70)
    print(
        f"BIORAG REQUEST: {drug}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # INDICATION
    # --------------------------------------------------------

    indication = (
        get_established_indication(
            drug
        )
    )

    if not indication:

        indication = (
            "Established indication "
            "not available."
        )

    print()
    print(
        "Established indication:"
    )

    print(
        indication
    )

    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    print()
    print(
        f"Retrieving up to "
        f"{RETRIEVAL_POOL} documents..."
    )

    try:

        retrieved = retrieve_documents(

            drug,

            question

        )

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )

    print(
        f"Retrieved: "
        f"{len(retrieved)}"
    )

    # --------------------------------------------------------
    # STRICT DRUG FILTER
    # --------------------------------------------------------

    drug_documents = []

    rejected_documents = []

    for document in retrieved:

        if document_is_about_drug(

            document,

            drug

        ):

            drug_documents.append(
                document
            )

        else:

            rejected_documents.append(
                document
            )

    print(
        f"Actual drug documents: "
        f"{len(drug_documents)}"
    )

    print(
        f"Rejected other-drug documents: "
        f"{len(rejected_documents)}"
    )

    # --------------------------------------------------------
    # ESTABLISHED FILTER
    # --------------------------------------------------------

    established_documents = []

    repurposing_documents = []

    for document in drug_documents:

        if is_established(

            document,

            indication

        ):

            established_documents.append(
                document
            )

        else:

            repurposing_documents.append(
                document
            )

    # Sort
    repurposing_documents.sort(

        key=lambda d:
            float(
                d.get(
                    "vector_score",
                    0
                )
            ),

        reverse=True

    )

    repurposing_documents = (
        repurposing_documents[
            :top_k
        ]
    )

    print(
        f"Established documents: "
        f"{len(established_documents)}"
    )

    print(
        f"Repurposing documents: "
        f"{len(repurposing_documents)}"
    )

    # --------------------------------------------------------
    # NO EVIDENCE
    # --------------------------------------------------------

    if not repurposing_documents:

        return {

            "success":
                True,

            "drug":
                drug,

            "question":
                question,

            "established_indication":
                indication,

            "retrieved_documents":
                len(retrieved),

            "drug_relevant_documents":
                len(drug_documents),

            "established_documents":
                len(established_documents),

            "repurposing_documents":
                0,

            "evidence":
                [],

            "answer":
                (
                    "# Repurposing Evidence\n\n"
                    "The current BioRAG corpus "
                    "did not retrieve sufficient "
                    "non-indication evidence "
                    f"for {drug}.\n\n"
                    "This does NOT prove that "
                    "no repurposing evidence "
                    "exists. It means the current "
                    "indexed evidence did not "
                    "produce a suitable signal."
                )

        }

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    context = build_context(
        repurposing_documents
    )

    print(
        f"LLM context size: "
        f"{len(context)} characters"
    )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    prompt = build_prompt(

        drug,

        question,

        indication,

        context

    )

    try:

        answer = generate_answer(
            prompt
        )

    except Exception as e:

        raise HTTPException(

            status_code=503,

            detail=str(e)

        )

    # --------------------------------------------------------
    # STRUCTURED EVIDENCE
    # --------------------------------------------------------

    evidence = []

    for document in repurposing_documents:

        evidence.append({

            "document_id":
                document.get(
                    "document_id"
                ),

            "type":
                format_value(
                    get_metadata(
                        document,
                        "type"
                    )
                ),

            "drug":
                format_value(
                    get_metadata(
                        document,
                        "drug"
                    )
                ),

            "disease":
                format_value(
                    get_metadata(
                        document,
                        "disease"
                    )
                ),

            "conditions":
                format_value(
                    get_metadata(
                        document,
                        "conditions"
                    )
                ),

            "title":
                format_value(
                    get_metadata(
                        document,
                        "title"
                    )
                ),

            "phase":
                format_value(
                    get_metadata(
                        document,
                        "phase"
                    )
                ),

            "status":
                format_value(
                    get_metadata(
                        document,
                        "status"
                    )
                ),

            "trial_id":
                format_value(
                    get_metadata(
                        document,
                        "trial_id"
                    )
                ),

            "nct_id":
                format_value(
                    get_metadata(
                        document,
                        "nct_id"
                    )
                ),

            "vector_score":
                document.get(
                    "vector_score"
                ),

            "evidence_level":
                evidence_level(
                    document
                )

        })

    # --------------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------------

    return {

        "success":
            True,

        "drug":
            drug,

        "question":
            question,

        "established_indication":
            indication,

        "retrieved_documents":
            len(retrieved),

        "drug_relevant_documents":
            len(drug_documents),

        "established_documents":
            len(established_documents),

        "repurposing_documents":
            len(repurposing_documents),

        "evidence":
            evidence,

        "answer":
            answer

    }


# ============================================================
# DIRECT RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "api.main:app",

        host="0.0.0.0",

        port=8000,

        reload=True

    )