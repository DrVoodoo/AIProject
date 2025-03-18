import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent

from dotenv import load_dotenv
import os

load_dotenv()

df = pd.read_csv("LivsmedelsDB_20241205_202503171443.csv")

engine = create_engine("sqlite:///nutrition.db")
df.to_sql("LivsmedelsDB_20241205_202503171443", engine, index=False, if_exists="replace")

db = SQLDatabase(engine=engine)

llm = ChatOpenAI(
   model_name="gpt-4o-mini",
   temperature=0.6,
   api_key=os.getenv("OPENAI_API_KEY")
)

agent = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)


if __name__ == "__main__":
   #result = agent_executor.invoke({"input": "what's the food item with the highest protein"})
   #result = agent_executor.invoke({"input": "vilka livsmedel innehåller ett namn med banan"})
   #result = agent_executor.invoke({"input": "visa innehållet för 'Banan kokbanan'"})
   #result = agent.invoke({"input": "vilka kolumner för ett livsmedel finns"})
   
   result = agent.invoke(
      {"input": "visa innehållet för 'Banan kokbanan'"},
      config={"configurable": {"thread_id": "default"}}
   )
   
   print('done')
   print(result)
   print(result["output"])