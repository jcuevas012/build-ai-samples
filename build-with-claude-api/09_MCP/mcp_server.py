from mcp.server.fastmcp import FastMCP
from pydantic import Field
from mcp.server.fastmcp.prompts import base

mcp = FastMCP("DocumentMCP", log_level="ERROR")


docs = {
    "deposition.md": "This deposition covers the testimony of Angela Smith, P.E.",
    "report.pdf": "The report details the state of a 20m condenser tower.",
    "financials.docx": "These financials outline the project's budget and expenditures.",
    "outlook.pdf": "This document presents the projected future performance of the system.",
    "plan.md": "The plan outlines the steps for the project's implementation.",
    "spec.txt": "These specifications define the technical requirements for the equipment.",
}

@mcp.tool(
          name="read_doc",
          description="Reads the content of a document given its ID"
          )
def read_doc(doc_id: str = Field(description="The ID of the document to read")) -> str:
    if doc_id not in docs:
        raise ValueError(f"Document with id '{doc_id}' not found.")
    return docs[doc_id]


@mcp.tool(
          name="edit_doc",
          description="Edits the content of a document given its ID"
          )
def edit_doc(
    doc_id: str = Field(description="The ID of the document to edit"),
    old_content: str = Field(description="The old content of the document"),
    new_content: str = Field(description="The new content for the document")) -> str:
    if doc_id not in docs:
        raise ValueError(f"Document with id '{doc_id}' not found.")
    
    docs[doc_id] = docs[doc_id].replace(old_content, new_content)


@mcp.resource(
    name="list_docs",
    uri="docs://documents",
    mime_type="application/json",
)
def list_docs() -> list[str]:
    return list(docs.keys())


@mcp.resource(
    name="get_doc",
    uri="docs://documents/{doc_id}",
    mime_type="text/plain",
)
def get_doc(doc_id: str) -> str:
    if doc_id not in docs:
        raise ValueError(f"Document with id '{doc_id}' not found.")
    return docs[doc_id]

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
                Use the 'edit_doc' tool to edit the document. After the document has been reformatted...
                """
    return [
        base.UserMessage(prompt)
    ]



if __name__ == "__main__":
    mcp.run(transport="stdio")
