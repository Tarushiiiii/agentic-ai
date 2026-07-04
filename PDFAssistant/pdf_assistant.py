import os
from typing import Optional, List

import typer
from dotenv import load_dotenv

from agno.agent import Agent
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.groq import Groq
from agno.db.postgres import PostgresDb
from agno.vectordb.pgvector import PgVector

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

embedder = OllamaEmbedder(id="nomic-embed-text", dimensions=768)

knowledge = Knowledge(
    vector_db=PgVector(
      table_name="recipes",
      db_url=db_url, 
      embedder=embedder
    ),    
)

db = PostgresDb(db_url=db_url, session_table="pdf_assistant_sessions")

def pdf_assistant(new: bool = False, user: str = "user")-> None:
  session_id: Optional[str] = None
 
  if not new:
      existing_sessions = db.get_sessions(user_id=user)
      existing_ids: List[str] = [s.session_id for s in existing_sessions]
      if existing_ids:
          choice = typer.prompt(
              f"Existing sessions for '{user}': {existing_ids}\n"
              "Enter a session ID to resume, or leave blank for a new one",
              default="",
              show_default=False,
          )
          if choice:
              if choice in existing_ids:
                  session_id = choice
              else:
                  typer.echo("Unknown session ID — starting a new session instead.")
  
  agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    session_id=session_id,
    user_id=user,
    knowledge=knowledge,
    db=db,
    search_knowledge=True,
    read_chat_history=True,
    add_history_to_context=True
  )
  
  if session_id is None:
    session_id = agent.session_id
    typer.echo(f"Started Session: {session_id}\n")
  else:
    typer.echo(f"Resuming Session: {session_id}\n")
  
  agent.cli_app(markdown=True)
 
if __name__ == "__main__":
  knowledge.insert_many(urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"])
  typer.run(pdf_assistant)
