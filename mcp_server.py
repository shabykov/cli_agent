import re
from typing import Annotated

from mcp.server.mcpserver import MCPServer, Context, Resolve, Sample
from mcp.server.mcpserver.prompts import base
from mcp.types import CreateMessageResult, SamplingMessage, TextContent
from pydantic import Field

mcp = MCPServer("DocumentMCP", log_level="ERROR")

docs = {
    "deposition.md": "This deposition covers the testimony of Angela Smith, P.E.",
    "report.pdf": "The report details the state of a 20m condenser tower.",
    "financials.docx": "These financials outline the project's budget and expenditures.",
    "outlook.pdf": "This document presents the projected future performance of the system.",
    "plan.md": "The plan outlines the steps for the project's implementation.",
    "spec.txt": "These specifications define the technical requirements for the equipment.",
}


# read a doc
@mcp.tool(
    name="read_doc_contents",
    description="Read the contents of a document and return it as a string."
)
def read_document(
        doc_id: str = Field(description="Id of the document to read")
):
    if doc_id not in docs:
        raise ValueError(f"Doc with id {doc_id} not found")

    return docs[doc_id]


# edit a doc
@mcp.tool(
    name="edit_document",
    description="Edit a document by replacing a string in the documents content with a new string."
)
def edit_document(
        doc_id: str = Field(description="Id of the document that will be edited"),
        old_str: str = Field(description="The text to replace. Must match exactly, including whitespace."),
        new_str: str = Field(description="The new text to insert in place of the old text.")
):
    if doc_id not in docs:
        raise ValueError(f"Doc with id {doc_id} not found")

    docs[doc_id] = docs[doc_id].replace(old_str, new_str)


# return all doc id's
@mcp.resource(
    "docs://documents",
    mime_type="application/json"
)
def list_docs() -> list[str]:
    return list(docs.keys())


# return the contents of a particular doc
@mcp.resource(
    "docs://documents/{doc_id}",
    mime_type="text/plain"
)
def fetch_doc(doc_id: str) -> str:
    if doc_id not in docs:
        raise ValueError(f"Doc with id {doc_id} not found")
    return docs[doc_id]


# rewrite a doc in markdown format
@mcp.prompt(
    name="format",
    description="Rewrites the contents of the document in Markdown format."
)
def format_document(
        doc_id: str = Field(description="Id of the document to format")
) -> list[base.Message]:
    prompt = f"""
        Your goal is to reformat a document to be written with markdown syntax.
        
        The id of the document you need to reformat is:
        
        {doc_id}
        
        
        Add in headers, bullet points, tables, etc as necessary. Feel free to add in extra formatting.
        Use the 'edit_document' tool to edit the document. After the document has been reformatted...
    """

    return [
        base.UserMessage(prompt)
    ]


def sampling_request(prompt: str) -> Sample:
    return Sample(
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(type="text", text=prompt)
            )
        ],
        max_tokens=4000,
        system_prompt="You are a helpful research assistant.",
    )


def sampled_text(result: CreateMessageResult) -> str:
    if result.content.type == "text":
        return result.content.text
    else:
        raise ValueError("Sampling failed")


# resolver: runs before the tool body, the framework sends the Sample to the client
async def request_summary(text_to_summarize: str, ctx: Context) -> Sample:
    await ctx.report_progress(10, 100, "About to do summarize...")
    return sampling_request(f"""
        Please summarize the following text:
        {text_to_summarize}
    """)


@mcp.tool(
    name="summarize",
    description="Read the contents of a document and summarize and return it as a string."
)
async def summarize(
        summary: Annotated[CreateMessageResult, Resolve(request_summary)],
        text_to_summarize: str = Field(description="Text of the document"),
        *,
        ctx: Context
):
    await ctx.report_progress(90, 100, "Writing summary...")
    return sampled_text(summary)


# resolver: searches the docs and asks the client to write a report, None if nothing found
async def request_report(topic: str, ctx: Context) -> Sample | None:
    await ctx.report_progress(20, 100, "About to do research...")
    sources = await do_research(topic)
    if not sources:
        return None

    sources_text = "\n".join(f"- {source}" for source in sources)
    return sampling_request(f"""
        Write a short research report on the topic: {topic}
        Base it on the following sources:
        {sources_text}
    """)


@mcp.tool(
    name="research",
    description="Research a given topic"
)
async def research(
        report: Annotated[CreateMessageResult | None, Resolve(request_report)],
        topic: str = Field(description="Topic to research"),
        *,
        context: Context
):
    await context.report_progress(70, 100, "Writing report...")
    if report is None:
        return f"No documents found for topic: {topic}"

    return sampled_text(report)


# search docs by keywords from the topic, most relevant first
async def do_research(topic: str) -> list[str]:
    keywords = {word for word in re.findall(r"\w+", topic.lower()) if len(word) >= 3}

    scored = []
    for doc_id, content in docs.items():
        text = f"{doc_id} {content}".lower()
        score = sum(1 for keyword in keywords if keyword in text)
        if score > 0:
            scored.append((score, f"{doc_id}: {content}"))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [source for _, source in scored]


if __name__ == "__main__":
    mcp.run(transport="stdio")
