from crewai import Agent, Task, Process, Crew
from langchain_google_genai import ChatGoogleGenerativeAI
import os

llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             verbose = True,
                             temperature = 0.5,
                             google_api_key="AIzaSyDCkRuQhULpNsCYRx-eNI-m7HTqO_pYNy0")




researcher = Agent(
    role='Researcher',
    goal = "Generate ideas for teaching someone new to the Subject.",
    backstory ="You are AI assistant to research and provide specific topics  or relevant information for teaching someone",
    verbose= True,
    allow_delegation=False,
    llm=llm
    
)

Writer = Agent(
    role='Writer',
    goal = "Write a clear and concise explanation of the topic using language and concepts understandable to a beginner. Aim to format in paragraphs.",
    backstory ="You are a writer that provides clear explanations of the topics researched to teach a beginner",
    verbose= True,
    allow_delegation=False,
    llm=llm
    
)

Examiner = Agent(
    role='Examiner',
    goal = "Create 2-3 multiple-choice questions based on the provided text and the questions should assess different levels of understanding and key points from the text.generate answer choices for each question, including one correct answer and several plausible distractors.",
    backstory ="You are a Examiner that generate questions based on the topic given.",
    verbose= True,
    allow_delegation=False,
    llm=llm
)


task1 = Task(description="Generate ideas for teaching someone new to the AI aubject.",
             agent=researcher)

task2 = Task(description="Use the Researcher’s ideas to Write a clear and concise explanation on some of the topics using language and concepts understandable to a beginner and format in paragraphs.",
             agent=Writer)

task3 = Task(description="Craft 2-3 test questions to evaluate understanding of the created text, along with the correct answers. In other words: test whether a student has fully understood the text.",
             agent=Examiner)

crew = Crew(
    agents =[researcher,Writer,Examiner],
    tasks=[task1, task2, task3],
    verbose=2,
    process=Process.sequential
)

result = crew.kickoff()

