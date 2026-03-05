print("*** Question Ansering Module Sucessfully Imported ***")

import subprocess
import re
import os 
import ollama
import time 
import subprocess
import requests
import torch
import threading
import concurrent.futures
import sys 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def allow_port(porta):
    try:
        comando = f'netstat -ano | findstr :{porta}'
        resultado = subprocess.run(comando, capture_output=True, text=True, shell=True)

        if not resultado.stdout.strip():
            print(f"Port {porta} is free.")
            return

        linhas = resultado.stdout.strip().split("\n")
        for linha in linhas:
            partes = re.split(r'\s+', linha)
            if len(partes) >= 5:
                pid = partes[-1]
                print(f"Process PID {pid} is using port {porta}. Trying to terminate process...")

                subprocess.run(f'taskkill /PID {pid} /F', shell=True)
                print(f"Process {pid} sucessfuly stoped.")

    except Exception as e:
        print(f"Erro ao liberar a porta {porta}: {e}")

def start_ollama():
    status = None
    controle = 0 
    while status is None and controle < 5: 
        process = subprocess.Popen(["ollama", "serve"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
       
    
        time.sleep(5) 
        status = process.poll() 
        controle += 1 

    print(f"Ollama running status: {process.poll()}")
   
    
    # Test if ollama is running
    try:
        answer = requests.get("http://localhost:11434/api/tags")
        if answer.status_code == 200:
            print("Ollama properly running!")
        else:
            print("WARNING: Ollama is not acessible.")
    except requests.exceptions.ConnectionError:
        print("ERROR\n There was an error while starting Ollama server.")
    
    
    ollama.pull("deepseek-r1:1.5b")

    return ollama 


def run_chatbot(question, file_content, prompt): 
    response = ollama.chat(model='deepseek-r1:1.5b', messages=[{'role': 'user', 
                                                                'content': f"Prompt: {prompt}\n\nQuestion:{question}\n\nFile Content \n{file_content}"
                                                               }],
                           options={
                                "temperature": 0                            }                        
                          )
    
    print("Chatbot response: \n", response)
    return response 

def run_chatbot_no_file(question, prompt): 
    response = ollama.chat(model='deepseek-r1:1.5b', messages=[{'role': 'user', 
                                                                'content': f"Prompt: {prompt}\n\nQuestion: {question}"
                                                               }])
    print("Chatbot response: \n", response)
    return response 

def thread_response():
    response = None
        
    chat_thread = threading.Thread(target=run_chatbot)
    chat_thread.start()
        
    # Espera até 60 segundos
    chat_thread.join(timeout=10) 
        
    if response is None:
        print("Timeout error")
    else:
        print(response['message']['content'])
    return response 

def get_content(files_to_read):
    contents = ''
    for file_to_read in files_to_read: 
        with open(file_to_read, "r") as file: 
            content = file.read()
            contents = contents + content 
            file.close()
        sys.stdout.flush()
        
    return contents

def run_futures(question, file_content, prompt):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_chatbot, question=question, file_content=file_content, prompt=prompt)
        response = future.result(timeout=60)
    return response 


def Question_Answering_Module(question, results_files, prompt, show_thinking = False):

    if results_files is not None: 
        file_content = get_content(results_files) 
        response = run_futures(question, file_content, prompt)
    else: 
        response = run_chatbot_no_file(question=question, prompt=prompt)
    
    if show_thinking == True: 
        # Displaying the response
        print(response['message']['content'])
    print("\n\n")
    return (response.message.content.split("</think>")[1])


def set_prompt(prompt):
    response = ollama.chat(
    model='deepseek-r1:1.5b', 
    messages=[
        {'role': 'system', 'content': prompt}]
    
)
