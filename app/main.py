from fastapi import FastAPI
from app.rag import load_vectorstore
from app.schemas import PatientResponse

app = FastAPI(title="AI Patient Safety Companion")

vectordb = load_vectorstore()

@app.post("/ask", response_model=PatientResponse)
def ask_question(question: str):

    docs = vectordb.similarity_search(question, k=3)
    context = " ".join([d.page_content for d in docs])

    # Basic guardrail
    if "diagnose" in question.lower():
        return PatientResponse(
            question=question,
            explanation="I cannot provide a medical diagnosis.",
            warnings=[],
            recommended_action="Please consult a healthcare professional.",
            disclaimer="This is not a medical diagnosis."
        )

    answer = f"Based on available medical instructions: {context}"

    return PatientResponse(
        question=question,
        explanation=answer,
        warnings=["If symptoms worsen, seek medical help."],
        recommended_action="Follow provided medical instructions carefully.",
        disclaimer="This is not a medical diagnosis. In emergencies, contact healthcare services."
    )
