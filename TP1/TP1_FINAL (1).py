# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 09:32:10 2022

@author: ferna
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import string
import scipy.io.wavfile as spiowf
from huffmancodec import HuffmanCodec
    

#Número de ocorrências de uma certa letra num ficheiro de texto

def texto(fonte,alfabeto):
    contador = np.zeros(len(alfabeto))
    for i in fonte:
        car = ord(i)
        if(65 <= car < 91):
            contador[car-65] += 1 #Letras maiusculas
        elif (97 <= car < 123):
            contador[car - 97 + 26] += 1 #Letras minusculas
    return contador

#Número de ocorrencias de uma certa letra num ficheiro de áudio 

def audio_imagem(fonte,alfabeto):
    contador = np.zeros(len(alfabeto))    #Matriz inicializada a 0
    for i in fonte:
        contador[i] += 1
    return contador

#Desenhar o histograma

def desenhar_histograma(alfabeto,contador):
    plt.figure(1)       #Abertura de um gráfico
    plt.bar(alfabeto,contador)    #Desenhar barras
    #Escrever nos eixos
    plt.xlabel('Simbolos')
    plt.ylabel('Frequência')
    
#Desenha gráficos sem barras, com pontos e tracejado 

def desenha_grafico(x,y,fig):
    plt.figure(fig)
    plt.plot(x,y,'k--')
    plt.plot(x,y,'ko')
    plt.xlabel('Janela')
    plt.ylabel('Informação Mútua')
    
#Cálculo da Probabilidade (Através da formúla)

def calculo_probabilidade(alfabeto,cont):
    probabilidade = np.zeros(len(cont))
    soma = sum(cont)
    for i in range(len(cont)):
        probabilidade[i] = cont[i]/soma
    return probabilidade

#Cálculo da Entropia (Através da formúla)

def entropia(alfabeto,cont):
    probabilidades = calculo_probabilidade(alfabeto, cont)
    probabilidade = []
    for i in range(len(probabilidades)):
        if(probabilidades[i]!=0):               
            probabilidade.append(round(probabilidades[i],6)) #Retira as probabilidades 0
    entropia = -sum(probabilidade * np.log2(probabilidade))
    return entropia, probabilidade

#Média e Variância ("Através da formula V(X) = E(X^2) - (E(X)^2)")

def media_variancia(probabilidade, comprimento):
    media = np.average(comprimento,axis=None,weights=probabilidade)
    print("Entropia Huffman: ",round(media,6))
    comprimento_array=[]
    for i in range(len(comprimento)):
        comprimento_array.append(comprimento[i]**2)    #Anexar o valor ao final da matriz E(x^2)
    variancia = (np.average(comprimento_array,axis=None,weights=probabilidade))-(media**2) #Calcula a média ponderada ao longo de um eixo específico
    print("Variância: ",round(variancia,6))    

#Entropia Conjunta num ficheiro de texto
   
def exercicio5_texto(fonte,alfabeto):
    indice=[]
    digit=''
    for i in range(len(fonte)):
        carater=ord(fonte[i]) #Converte para asciiz
        if(65 <= carater <= 90):
            digit = ord(fonte[i])-65 # Devolve a posição
        elif(97 <= carater <= 122):
            digit = ord(fonte[i])-97+(26)
             
        if(i%2==0):
             digit1=digit
        else:
            digit2=digit
            ind= digit1*len(alfabeto) + digit2
            print(digit1)
            indice.append(ind)
               
    entropia_conjunta(alfabeto,indice)
           
#Entropia conjunta num ficheiro de audio ou imagem 

def exercicio5_audioImagem(fonte,alfabeto):
    indice=[]   
    for i in range(len(fonte)):
        if(i%2==0):
            digit1=int(fonte[i])
        else:
            digit2=int(fonte[i])
            ind= digit1*len(alfabeto) + digit2
            indice.append(ind)     
      
    entropia_conjunta(alfabeto,indice)
    
#Entropia Conjunta(Auxiliar para não haver repetição de código)
    
def entropia_conjunta(alfabeto,indice):
    ocorrencia=np.zeros(len(alfabeto)**2)
    for i in range(len(indice)):
        ocorrencia[indice[i]] +=1
    alfabeto1=np.arange(0,len(alfabeto)**2)
    entropia_conj, probabilidade = entropia(alfabeto1,ocorrencia)
    entropia_conj = (entropia_conj)/2
    print("Entropia Conjunta: ",round(entropia_conj,6))
    
#Cálculo da probabilidade conjunta

def probabilidade_conjunta(alfabeto,query,target):
    prob=np.zeros((len(alfabeto),len(alfabeto)))
    for i in range(len(query)):
        prob[query[i]][target[i]] += 1
    prob=prob /len(query)
    return prob


#Simulação do Cálculo da informação Mútua do exercicio 6 alinea a)

def simuacao_informacaoMutua(query, target, alfabeto,passo):
    print("\nExercicio 6:\nAlínea a): ")
    resultado = informacaoMutua(alfabeto, query, target, passo);
    print("Infomação Mutua: ",resultado)

#Cálculo da informação mútua

def informacaoMutua(alfabeto,query,target,passo):
    p=0
    count_query= audio_imagem(query,alfabeto) #Cálcula o número de ocorrências do query (sinal a pesquisar, neste caso o numero de ocorrência da lista dada)
    probabilidade_query= calculo_probabilidade(alfabeto,count_query)
    array=[]
    informacoes_mutuas=[]
    while True:
        janela = target[p:len(query)+p]
        if(len(janela) != len(query)):
            break   #Significa que não irá ocorrer o "deslizamento"
        p += passo
        prob = probabilidade_conjunta(alfabeto,query,janela) #Fórmula p(y/x)
        contagem_janela = audio_imagem(janela,alfabeto) 
        probabilidade_janela = calculo_probabilidade(alfabeto,contagem_janela)
        informacao_mutua = 0
        for i in range(len(alfabeto)):          
            array=np.arange(len(query))
            array=prob[i]*(np.log2((prob[i]/(probabilidade_query[i]*probabilidade_janela))))          
            aux= np.logical_and(np.logical_not(np.isnan(array)),np.logical_not(np.isinf(array))) #Calcular o valor de verdade
            informacao_mutua +=sum(array[aux])
            
        informacoes_mutuas.append(round(informacao_mutua,4)) #Lista de resultados
        
    return informacoes_mutuas #(Array 1x41)

    
#Cálculo da informação mútua em relação aos targets

def exercicio6b(query,target1,target2):
    [fs,fonte] = spiowf.read(query)
    if(fonte.ndim>1):
        fonte = fonte[:,0]
    fonte = np.array(fonte)
    tipo = str(fonte.dtype)
    digit =''
    for elemento in tipo:
        if(elemento.isdigit()):
            digit+=elemento
    alfabeto = np.arange(0,2**int(digit))
    #Lê os ficheiros target
    [fs1,fonte1] = spiowf.read(target1)
    [fs2,fonte2] = spiowf.read(target2) 
    if(fonte1.ndim>1): #Dimensão do array superior a 1
        fonte1 = fonte1[:,0]
    fonte1 = np.array(fonte1)
    if(fonte2.ndim>1):
        fonte2 = fonte2[:,0]
    fonte2 = np.array(fonte2)
    passo= (len(fonte))//4
    #Cálcula a informação mutua dos targets
    infor_mutua_target1 = informacaoMutua(alfabeto,fonte,fonte1,passo)
    infor_mutua_target2= informacaoMutua(alfabeto,fonte,fonte2,passo)
    print("\nAlínea b):\n")
    print("Target01 - repeat.wav")
    print(infor_mutua_target1)
    print("Target02 - repeatNoise.wav")
    print(infor_mutua_target2)
    array=np.arange(0,len(infor_mutua_target1))
    desenha_grafico(array,infor_mutua_target1,2)
    desenha_grafico(array,infor_mutua_target2,3)
    
#Cálculo da informação mútua em relação aos targets

def exercicio6c(query,sons):
     [fs,fonte] = spiowf.read(query)
     if(fonte.ndim>1):
         fonte = fonte[:,0]
     fonte = np.array(fonte)
     tipo = str(fonte.dtype)
     digit =''
     for elemento in tipo:
        if(elemento.isdigit()):
            digit+=elemento
     alfabeto = np.arange(0,2**int(digit))
     informacoes=[]
     passo= len(fonte)//4
     print("\nAlinea c): \n")
     for i in range(7):
         [fs1,fonte1] = spiowf.read(sons[i])
         if(fonte1.ndim>1):
             fonte1 = fonte1[:,0]
         fonte1 = np.array(fonte1)
         informacoes.append(informacaoMutua(alfabeto,fonte,fonte1,passo))
     for i in range(len(informacoes)):
        print(sons[i][-10:])
        print(informacoes[i])
        aux= sorted(informacoes[i],reverse=True) #Ordenamos o vetor para depois ser mais fácil colocar por ordem decrescente
        array=np.arange(0,len(informacoes[i]))
        desenha_grafico(array,informacoes[i],i+4)
     decrescente=[]      
     for i in range(len(informacoes)):
        array=[max(informacoes[i]),sons[i][-10:]]
        decrescente.append(array) 
     aux = sorted(decrescente, key=lambda x: x[0], reverse=True)
     print("\nInformação mútua por ordem decrescente:\n")
     print(aux)
     
#Para evitar repetição de código no tratamento de dados(é comum a todos os tipos de ficheiro) (retorna a probabilidade, pois é necessária no ex4)
     
def aux_print_entropia(alfabeto,count):
    desenhar_histograma(alfabeto, count)
    entrop,probabilidade = entropia(alfabeto,count)
    print("Entropia: ",round(entrop,6)) 
    return probabilidade
    
#Função principal, onde são formecidos dados e são chamadas as funções
     
def tratamento_de_dados():
    #Declaração de dados que irão ser utilizados(Depende da localização dos ficheiros,a seguir encontra se a localização dos mesmo no ambiente onde o código foi criado)
    
    ficheiro = 'C:\\Users\\User\\Desktop\\DATA\\lyrics.txt'
    #ficheiro = 'C:\\Users\\User\\Desktop\\DATA\\MRIbin.bmp'
    #ficheiro = 'C:\\Users\\User\\Desktop\\DATA\\MRI.bmp'
    #ficheiro = 'C:\\Users\\User\\Desktop\\DATA\\landscape.bmp'
    #ficheiro = 'C:\\Users\\User\\Desktop\\DATA\\soundMono.wav'
    
    query = [2,6,4,10,5,9,5,8,0,8] #(array 1x10)  
    target = [6,8,9,7,2,4,9,9,4,9,1,4,8,0,1,2,2,6,3,2,0,7,4,9,5,4,8,5,2,7,8,0,7,4,8,5,7,4,3,2,2,7,3,5,2,7,4,9,9,6] #(array 1x50)
    alfabeto = [0,1,2,3,4,5,6,7,8,9,10]
    passo = 1
    
    query1="C:\\Users\\User\\Desktop\\DATA\\MI\\saxriff.wav"
    target1="C:\\Users\\User\\Desktop\\DATA\\MI\\target01 - repeat.wav"
    target2="C:\\Users\\User\\Desktop\\DATA\\MI\\target02 - repeatNoise.wav"
    
    sons=["C:\\Users\\User\\Desktop\\DATA\\MI\\Song01.wav","C:\\Users\\User\\Desktop\\DATA\\MI\\Song02.wav","C:\\Users\\User\\Desktop\\DATA\\MI\\Song03.wav","C:\\Users\\User\\Desktop\\DATA\\MI\\Song04.wav","C:\\Users\\User\\Desktop\\DATA\\MI\\Song05.wav","C:\\Users\\User\\Desktop\\DATA\\MI\\Song06.wav","C:\\Users\\User\\Desktop\\DATA\\MI\\Song07.wav"]

    aux = 0; #Decide que tipo fonte se está a trabalhar (.txt ou .wav/.bmp) (necessário para o ex5)
    
    print("\nExercicio 3: ")
    
    #Tratamento de dados .txt
    
    if(ficheiro.endswith('.txt')):
        with open(ficheiro) as f:
            fonte = f.read()
            fonte = list(fonte)
            comp = len (fonte)
            indice = 0
        for i in range(0,comp):
            if not (('a' <= fonte[indice] <= 'z') or ('A' <= fonte[indice] <= 'Z')):
                fonte.remove(fonte[indice])
            else:
                indice += 1
        alfabeto = list(string.ascii_uppercase) + list(string.ascii_lowercase)
        count = texto(fonte,alfabeto)
        probabilidade = aux_print_entropia(alfabeto, count)
        aux = 1
        
    #Tratamento de dados .bmp
    
    elif(ficheiro.endswith('.bmp')):
         fonte = mpimg.imread(ficheiro)
         if(fonte.ndim>2):
             fonte=fonte[:,:,0]
         fonte = np.array(fonte).flatten() #Retorna a copia do array colapsado
         tipo = str(fonte.dtype)
         digit = ''
         for elemento in tipo:
             if(elemento.isdigit()):
                 digit+=elemento
         alfabeto = np.arange(0,2**int(digit))
         count = audio_imagem(fonte, alfabeto)
         probabilidade = aux_print_entropia(alfabeto, count)
         
    #Tratamento de dados .wav      
    
    elif(ficheiro.endswith('.wav')):
        [fs,fonte] = spiowf.read(ficheiro)
        if(fonte.ndim>1):
            fonte = fonte[:,0]
        fonte = np.array(fonte)
        tipo = str(fonte.dtype)
        digit =''
        for elemento in tipo:
            if(elemento.isdigit()):
                digit+=elemento
        alfabeto = np.arange(0,2**int(digit))  
        count = audio_imagem(fonte,alfabeto)
        probabilidade = aux_print_entropia(alfabeto, count)
        
    
    codec = HuffmanCodec.from_data(fonte)
    symbols, lengths = codec.get_code_len() #Da bibioteca
    
    print("\nExercicio 4: ")
    media_variancia(probabilidade, lengths)
    
    if(aux==1):
       print("\nExercicio 5: ")
       exercicio5_texto(fonte, alfabeto)
    else:
       print("\nExercicio 5: ")
       exercicio5_audioImagem(fonte, alfabeto)

    np.seterr(divide='ignore', invalid='ignore') #Ignora potenciais avisos
    simuacao_informacaoMutua(query, target, alfabeto,passo)
    exercicio6b(query1,target1,target2)
    exercicio6c(query1,sons)
tratamento_de_dados()       
                
                
            
    
    

    