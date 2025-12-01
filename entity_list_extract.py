import json
from tqdm import tqdm
import random
import re
from vllm import LLM, SamplingParams

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, top_p=0.9, max_tokens = 512)

# Create an LLM.
llm = LLM(model="Model path")  #Qwen3-8b

prompt_x = """
As a truthful and objective query specialist, your role is to craft precise queries for verifying the accuracy of provided answers.In the #Thought-k# section, start by identifying indirect reference not indicated in both the question and the answer, guiding the focus of your initial queries. Then, scrutinize each detail in the answer to determine what needs verification and propose the corresponding #Query-k#. For information not indicated in both, initiate with a direct query and a rephrased broader context version in brackets. For details given in the answer, pose a general query without specifying the key entity for a wider context. Your goal is to methodically gather clear, relevant information to assess the answer's correctness.

#Question#: Who composed the famous musical score for the 1977 space-themed movie in which the character Luke Skywalker first appeared?
#Answer#: Joy Williams composed the score for "Star Wars."
#Thought-1#: The first query should confirm whether "Star Wars" is the 1977 space-themed movie in which Luke Skywalker first appeared, as this is necessary to link the movie to the composer.
#Query-1#: Which 1977 space-themed movie featured the first appearance of the character Luke Skywalker?
#Knowledge-1#: ("Star Wars", was, 1977 space-themed movie), (Luke Skywalker, first appeared in, "Star Wars")
#Thought-2#: Having established "Star Wars" as the relevant movie, the next step is to verify if Joy Williams, as mentioned in the answer, was indeed the composer of its score.
#Query-2#: Who composed the score for "Star Wars"?
#Knowledge-2#: (John Williams, composed, "Star Wars" score)
#Thought-3#: Found one wrong detail, we do not need further query.

#Question#: Which filmmaker was responsible for the 2010 fantasy film that prominently features the song "I See the Light"?
#Answer#: Christopher Nolan
#Thought-1#: The first query should identify the 2010 fantasy film that includes the song "I See the Light," leading to the filmmaker.
#Query-1#: Which 2010 fantasy film features the song "I See the Light"?
#Knowledge-1#: ("Tangled", is, 2010 fantasy film), ("I See the Light", featured in, "Tangled")
#Thought-2#: With "Tangled" identified, the next step is to determine if Christopher Nolan, as mentioned in the answer, directed it.
#Query-2#: Who directed the film "Tangled"?
#Knowledge-2#: (Nathan Greno and Byron Howard, directed, "Tangled")
#Thought-3#: Found one wrong detail, we do not need further query.

#Question#: Which Nobel Prize-winning physicist, known for his work on quantum theory, was born in the city that hosted the 1936 Summer Olympics?
#Answer#: Walther Meissner
#Thought-1#: The first query should identify the city that hosted the 1936 Summer Olympics, as this information is not indicated in either the question or the answer. Once identified, we can then establish a connection with the physicist's birthplace.
#Query-1#: In which city were the 1936 Summer Olympics held?
#Knowledge-1#: (1936 Summer Olympics, hosted in, Berlin)
#Thought-2#: Next, verify if Walther Meissner was born in Berlin, as it would confirm his connection to the city.
#Query-2#: In which city was Walther Meissner born?
#Knowledge-2#: (Walther Meissner, born in, Berlin)
#Thought-3#: After confirming Meissner's birthplace, investigate whether he was a Nobel Prize laureate, as mentioned in the answer.
#Query-3#: What award did Walther Meissner win?
#Knowledge-3#: No specific information is available.
#Thought-4#: Although we couldn't obtain specific information regarding Walther Meissner's Nobel Prize, let's proceed to examine whether Meissner is known for his contributions to quantum theory, as claimed in the answer. 
#Query-4#: In what area is Walther Meissner known for his work?
#Knowledge-4#: (Walther Meissner, known for, work in low-temperature physics)
#Thought-5#: Found one wrong detail, we do not need further query.

#Question#: "Can you provide the names of individuals who were nominated for the Grammy Award for Best Musical Theater Album and play the piano?"
#Answer#: "The individuals who meet the given criteria are David Foster and Cy Coleman."
#Thought-1#: The first query should confirm if David Foster and Cy Coleman were indeed nominated for the Grammy Award for Best Musical Theater Album.
#Query-1#: Who were the nominees for the Grammy Award for Best Musical Theater Album?
#Knowledge-1#: (David Foster, nominated for, Grammy Award for Best Musical Theater Album), (Cy Coleman, nominated for, Grammy Award for Best Musical Theater Album)
#Thought-2#: Having confirmed their nominations, the next step is to verify if both individuals play the piano, as mentioned in the answer.
#Query-2#: Are David Foster and Cy Coleman pianists?
#Knowledge-2#: (David Foster, is a, pianist), (Cy Coleman, was a, pianist)
#Thought-3#: All the necessary information to judge the correctness of the answer has been obtained, so the query process can now be concluded.

#Question#: "Based on the population data from 1971, which country had the highest population, and which city had the lowest population among Liberia, Mannheim, Ireland, Samoa, and Sierra Leone?"
#Answer#: "Among the given countries and cities, *Sierra Leone* had the highest population in 1971, with a population of 3,000,000. *Mannheim* had the lowest population among the given countries and cities, with a population of 100,000. *Liberia* had a population of 1,200,000, *Ireland* had a population of 2,500,000, and *Samoa* had a population of 200,000."
#Thought-1#: The first query should confirm the population data for each of the specified countries and cities in 1971.
#Query-1#: What was the population of Liberia, Mannheim, Ireland, Samoa, and Sierra Leone in 1971?
#Knowledge-1#: (Liberia, population in 1971, 1,200,000), (Mannheim, population in 1971, 100,000), (Ireland, population in 1971, 2,500,000), (Samoa, population in 1971, 200,000), (Sierra Leone, population in 1971, 3,000,000)
#Thought-2#: Having confirmed the population data, the next step is to verify the claim that Sierra Leone had the highest population and Mannheim had the lowest population among the given countries and cities.
#Query-2#: Which country and city had the highest and lowest populations among Liberia, Mannheim, Ireland, Samoa, and Sierra Leone in 1971?
#Knowledge-2#: (Sierra Leone, had the highest population in 1971), (Mannheim, had the lowest population in 1971)
#Thought-3#: All the necessary information to judge the correctness of the answer has been obtained, so the query process can now be concluded.

#Question#: "What is the occupation of the person who won the Satellite Award for Best Original Screenplay?"
#Answer#: "The person who won the Satellite Award for Best Original Screenplay, George Clooney, is a *director*."
#Thought-1#: The first query should confirm if George Clooney won the Satellite Award for Best Original Screenplay.
#Query-1#: Who won the Satellite Award for Best Original Screenplay?
#Knowledge-1#: (George Clooney, won the Satellite Award for Best Original Screenplay)
#Thought-2#: Having confirmed George Clooney's win, the next step is to verify his occupation as a director, as mentioned in the answer.
#Query-2#: What is George Clooney's occupation?
#Knowledge-2#: (George Clooney, occupation, director)
#Thought-3#: All the necessary information to judge the correctness of the answer has been obtained, so the query process can now be concluded.

#Question#: "What is the name of the administrative territorial entity that contains Key West?"
#Answer#: "The administrative territorial entity that contains Key West is *Duval County*."
#Thought-1#: The first query should confirm if Duval County is indeed the administrative territorial entity that contains Key West.
#Query-1#: What county is Key West in?
#Knowledge-1#: (Key West, is in, Monroe County)
#Thought-2#: Found one wrong detail, we do not need further query.

#Question#: "Can you provide the names of rock genres that both Carlos Santana and Deep Purple are famous for?"
#Answer#: "The rock genres that both Carlos Santana and Deep Purple are famous for are \"hard rock,\" \"psychedelic rock,\" and \"blues rock.\""
#Thought-1#: The first query should confirm if Carlos Santana and Deep Purple are indeed famous for the rock genres mentioned in the answer.
#Query-1#: What rock genres is Carlos Santana famous for?
#Knowledge-1#: (Carlos Santana, is famous for, hard rock),(Carlos Santana, is famous for, psychedelic rock),(Carlos Santana, is famous for, blues rock)
#Thought-2#: Having confirmed Carlos Santana's rock genres, the next step is to verify if Deep Purple is also famous for these genres.
#Query-2#: What rock genres is Deep Purple famous for?
#Knowledge-2#: (Deep Purple, is famous for, hard rock),(Deep Purple, is famous for, psychedelic rock),(Deep Purple, is famous for, blues rock)
#Thought-3#: All the necessary information to judge the correctness of the answer has been obtained, so the query process can now be concluded.

#Question#: "Which films were released in both Hungary and North America?"
#Answer#: "The films that were released in both Hungary and North America are \"The Godfather\" and \"Gone with the Wind.\""
#Thought-1#: The first query should confirm if "The Godfather" and "Gone with the Wind" were indeed released in both Hungary and North America.
#Query-1#: Which films were released in both Hungary?
#Knowledge-1#: ("The Godfather", released in, Hungary), ("Gone with the Wind", released in, Hungary)
#Thought-2#: Having confirmed the release of "The Godfather" and "Gone with the Wind" in both locations, the next step is to verify if these are the only films to meet this criterion.
#Query-2#: Which films were released in both North America?
#Knowledge-2#: ("The Godfather", released in, North America), ("Gone with the Wind", released in, North America)
#Thought-3#: All the necessary information to judge the correctness of the answer has been obtained, so the query process can now be concluded.

#Question#: "Can you provide the names of films that are based on literature and have Brad Dourif as a cast member?"
#Answer#: "The films that meet the given criteria are \"Dune,\" \"One Flew Over the Cuckoo's Nest,\" \"Ragtime,\" \"The Lord of the Rings: The Two Towers,\" and \"The Lord of the Rings: The Return of the King.\""
#Thought-1#: The first query should confirm if the films listed in the answer are indeed based on literature.
#Query-1#: Which of the following films are based on literature: "Dune," "One Flew Over the Cuckoo's Nest," "Ragtime," "The Lord of the Rings: The Two Towers," and "The Lord of the Rings: The Return of the King"?
#Knowledge-1#: ("Dune", is based on, literature), ("One Flew Over the Cuckoo's Nest", is based on, literature), ("Ragtime", is based on, literature), ("The Lord of the Rings: The Two Towers", is based on, literature), ("The Lord of the Rings: The Return of the King", is based on, literature)
#Thought-2#: Having confirmed that all listed films are based on literature, the next step is to verify if Brad Dourif was a cast member in each of these films.
#Query-2#: Which of the following films have Brad Dourif as a cast member: "Dune," "One Flew Over the Cuckoo's Nest," "Ragtime," "The Lord of the Rings: The Two Towers," and "The Lord of the Rings: The Return of the King"?
#Knowledge-2#: ("Dune",cast member, Brad Dourif), ("One Flew Over the Cuckoo's Nest", cast member, Brad Dourif), ("Ragtime", cast member, Brad Dourif), ("The Lord of the Rings: The Two Towers", cast member, Brad Dourif), ("The Lord of the Rings: The Return of the King", cast member, Brad Dourif)
#Thought-3#: All the necessary information to judge the correctness of the answer has been obtained, so the query process can now be concluded.


Please ensure that all queries are direct, clear, and explicitly relate to the specific context provided in the question and answer. Avoid crafting indirect or vague questions like 'What is xxx mentioned in the question?' Additionally, be mindful not to combine multiple details needing verification in one query. Address each detail separately to avoid ambiguity and ensure focused, relevant responses. Besides, follow the structured sequence of #Thought-k#, #Query-k#, #Knowledge-k# to systematically navigate through your verification process.

#Question#: {}
#Answer#: {}
"""

def reason_generation(question,answer,prompt):
    prompts = prompt.format(question,answer)
    outputs = llm.generate(prompts, sampling_params,use_tqdm=False)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # print(f"Prompt: {prompt!r}, {generated_text!r}")
        return generated_text

def get_data(path):
    test_objects = []
    with open(path, 'r') as file:
        for line in file:
            try:
                json_object = json.loads(line)
                test_objects.append(json_object)
            except json.JSONDecodeError:
                print(f"Failed to decode JSON on line: {line}")
    return test_objects


def extract_first_question_pairs(text):
    # Split the list of text behaviors
    lines = text.split('\n')

    result = []

    # Flag bit: Whether the first Question has been reached
    reached_first_question = False

    for line in lines:
        # Skip blank lines
        if not line.strip():
            continue

        # Check if you have reached the first Question
        if line.startswith('#Question#') and not reached_first_question:
            reached_first_question = True
            continue  # Skip the Question line itself

        # Stop processing after reaching the first Question
        if reached_first_question:
            break

        # Match the Query and Knowledge rows
        query_match = re.match(r'#Query-(\d+)#: (.*)', line)
        knowledge_match = re.match(r'#Knowledge-(\d+)#: \((.*)\)', line)

        if query_match:
            result.append(f'#Query-{query_match.group(1)}#: {query_match.group(2)}')
        elif knowledge_match:
            result.append(f'#Knowledge-{knowledge_match.group(1)}#: ({knowledge_match.group(2)})')

    return '\n'.join(result)

def extract_q_ent(text):
    result = []
    result_tail = []
    lines = text.split('\n')
    for line in lines:
        knowledge_match = re.match(r'#Knowledge-(\d+)#: (.*)', line)
        if knowledge_match:
            triples = knowledge_match.group(2).split("), ")

            # Extract the entities of each triplet
            for triple in triples:
                parts = triple.strip("()").split(", ")
                if parts:
                    if parts[0].strip('"') not in result:
                        result.append(parts[0].strip('"'))
                    if len(parts) > 2 and parts[len(parts)-1] not in result:
                        result_tail.append(parts[len(parts)-1])
    for ent in result_tail:
        ent1 = ent.strip('"')
        if ent1 not in result:
            result.append(ent)
    return result

def extract_ent(data):
    subgraph_extract = []
    for i in tqdm(range(len(data[0]))):
        question = data[0][i]['question']
        answer = data[0][i]['answer']
        res = reason_generation(question, answer, prompt_x)
        qk = extract_first_question_pairs(res)
        entities = extract_q_ent(qk)
        subgraph_extract.append(entities)

if __name__ == '__main__':
    data_objects = get_data('data path')
    extract_ent = extract_ent(data_objects)
    with open('ent list save path', 'w', encoding='utf-8') as file:
        json.dump(extract_ent, file) # save the extract entity
