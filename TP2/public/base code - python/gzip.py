
import sys
from huffmantree import HuffmanTree
import numpy as np


class GZIPHeader:
    ''' class for reading and storing GZIP header fields '''

    ID1 = ID2 = CM = FLG = XFL = OS = 0
    MTIME = []
    lenMTIME = 4
    mTime = 0

    # bits 0, 1, 2, 3 and 4, respectively (remaining 3 bits: reserved)
    FLG_FTEXT = FLG_FHCRC = FLG_FEXTRA = FLG_FNAME = FLG_FCOMMENT = 0

    # FLG_FTEXT --> ignored (usually 0)
    # if FLG_FEXTRA == 1
    XLEN, extraField = [], []
    lenXLEN = 2

    # if FLG_FNAME == 1
    fName = ''  # ends when a byte with value 0 is read

    # if FLG_FCOMMENT == 1
    fComment = ''   # ends when a byte with value 0 is read

    # if FLG_HCRC == 1
    HCRC = []

    def read(self, f):
        ''' reads and processes the Huffman header from file. Returns 0 if no error, -1 otherwise '''

        # ID 1 and 2: fixed values
        self.ID1 = f.read(1)[0]
        if self.ID1 != 0x1f:
            return -1  # error in the header

        self.ID2 = f.read(1)[0]
        if self.ID2 != 0x8b:
            return -1  # error in the header

        # CM - Compression Method: must be the value 8 for deflate
        self.CM = f.read(1)[0]
        if self.CM != 0x08:
            return -1  # error in the header

        # Flags
        self.FLG = f.read(1)[0]

        # MTIME
        self.MTIME = [0]*self.lenMTIME
        self.mTime = 0
        for i in range(self.lenMTIME):
            self.MTIME[i] = f.read(1)[0]
            self.mTime += self.MTIME[i] << (8 * i)

        # XFL (not processed...)
        self.XFL = f.read(1)[0]

        # OS (not processed...)
        self.OS = f.read(1)[0]

        # --- Check Flags
        self.FLG_FTEXT = self.FLG & 0x01
        self.FLG_FHCRC = (self.FLG & 0x02) >> 1
        self.FLG_FEXTRA = (self.FLG & 0x04) >> 2
        self.FLG_FNAME = (self.FLG & 0x08) >> 3
        self.FLG_FCOMMENT = (self.FLG & 0x10) >> 4

        # FLG_EXTRA
        if self.FLG_FEXTRA == 1:
            # read 2 bytes XLEN + XLEN bytes de extra field
            # 1st byte: LSB, 2nd: MSB
            self.XLEN = [0]*self.lenXLEN
            self.XLEN[0] = f.read(1)[0]
            self.XLEN[1] = f.read(1)[0]
            self.xlen = self.XLEN[1] << 8 + self.XLEN[0]

            # read extraField and ignore its values
            self.extraField = f.read(self.xlen)

        def read_str_until_0(f):
            s = ''
            while True:
                c = f.read(1)[0]
                if c == 0:
                    return s
                s += chr(c)

        # FLG_FNAME
        if self.FLG_FNAME == 1:
            self.fName = read_str_until_0(f)

        # FLG_FCOMMENT
        if self.FLG_FCOMMENT == 1:
            self.fComment = read_str_until_0(f)

        # FLG_FHCRC (not processed...)
        if self.FLG_FHCRC == 1:
            self.HCRC = f.read(2)

        return 0


class GZIP:
    ''' class for GZIP decompressing file (if compressed with deflate) '''

    gzh = None
    gzFile = ''
    fileSize = origFileSize = -1
    numBlocks = 0
    f = None

    bits_buffer = 0
    available_bits = 0

    def __init__(self, filename):
        self.gzFile = filename
        self.f = open(filename, 'rb')
        self.f.seek(0, 2)
        self.fileSize = self.f.tell()
        self.f.seek(0)

    def decompress(self):
        ''' main function for decompressing the gzip file with deflate algorithm '''

        numBlocks = 0

        # get original file size: size of file before compression
        origFileSize = self.getOrigFileSize()
        print(origFileSize)

        # read GZIP header
        error = self.getHeader()
        if error != 0:
            print('Formato invalido!')
            return

        # show filename read from GZIP header
        print(self.gzh.fName)

        # MAIN LOOP - decode block by block
        BFINAL = 0
        while not BFINAL == 1:

            BFINAL = self.readBits(1)

            BTYPE = self.readBits(2)

            if BTYPE != 2:
                print('Error: Block %d not coded with Huffman Dynamic coding' %
                      (numBlocks+1))
                return
            #START HERE
            
            print("\nExercicio 1\n")
            
            
            HLIT = self.readBits(5)
            HDIST = self.readBits(5)
            HCLEN = self.readBits(4)
            
            print("HTIL:" + str( HLIT))
            print("HDIST:" + str(HDIST))
            print("HCLEN:" + str(HCLEN))
            
            
            HLIT += 257
            HDIST += 1
            HCLEN += 4
            ordem = [16, 17, 18, 0, 8, 7, 9, 6, 10,
                     5, 11, 4, 12, 3, 13, 2, 14, 1, 15]
            
            print("\nExercicio 2\n")
            
            n_array = self.exercicio2(HCLEN,ordem)
            print(n_array)
            
            print("\nExercicio 3\n") 
            
            codigo_arvore = self.exercicio3(n_array, ordem)
            print(codigo_arvore)
                
            print("\nExercicio 4\n")
            
            self.inserirTree(codigo_arvore)
            
        
            print("\nCodigos referentes ao alfabeto dos literais/tamanhos\n")
            code = self.exercicio4_5(HLIT, codigo_arvore)
            print(code)
            
            print("\nExercicio 5\n")
            
            code1 = self.exercicio4_5(HDIST, codigo_arvore)
            print(code1)
    
            
            print("\nExercicio 6\n")
            
            code_literal, code_dist, treeCodeLiterais, treeCodeDist = self.exercicio6(code1,code)
            print(code_literal)
            print("\n")
            print(code_dist)
         
            print("\nExercicio 7\n")
            
            
            vetor, texto = self.exercicio7(treeCodeLiterais,treeCodeDist)
            print(texto)
            
            print("\nExercicio 8\n")
            
            FileName = GZIPHeader.fName
            print("Nome do ficheiro = " + str(FileName))
            Ficheiro = open(FileName,'w')
            Ficheiro.write(texto)
            Ficheiro.close()
            
            print("\n")
            
           
            # update number of blocks read
            numBlocks += 1

        # close file

        self.f.close()
        print("End: %d block(s) analyzed." % numBlocks)

    def getOrigFileSize(self):
        ''' reads file size of original file (before compression) - ISIZE '''

        # saves current position of file pointer
        fp = self.f.tell()

        # jumps to end-4 position
        self.f.seek(self.fileSize-4)

        # reads the last 4 bytes (LITTLE ENDIAN)
        sz = 0
        for i in range(4):
            sz += self.f.read(1)[0] << (8*i)

        # restores file pointer to its original position
        self.f.seek(fp)

        return sz

    def getHeader(self):
        ''' reads GZIP header'''

        self.gzh = GZIPHeader()
        header_error = self.gzh.read(self.f)
        return header_error

    def readBits(self, n, keep=False):
        ''' reads n bits from bits_buffer. if keep = True, leaves bits in the buffer for future accesses '''

        while n > self.available_bits:
            self.bits_buffer = self.f.read(1)[0] << self.available_bits | self.bits_buffer
            self.available_bits += 8

        mask = (2**n)-1
        value = self.bits_buffer & mask

        if not keep:
            self.bits_buffer >>= n
            self.available_bits -= n

        return value
    
    def exercicio2(self,HCLEN,ordem):
        array = np.zeros(len(ordem),int)
        n_array = np.zeros(len(ordem),int)
        for i in range(HCLEN):
            array[i] = self.readBits(3)  
            for i in range(len(array)):   #Ordenação do vetor, fácil compreensão
                n_array[ordem[i]] = array[i]
        return n_array
    
    def exercicio3(self,n_array,ordem):
        #Passo 1 (DOC 2)
        
        numeros = []
        for i in range (len(n_array)):
            if n_array[i] not in numeros:
                numeros.append(n_array[i])  
        numeros = np.asarray(sorted(numeros)) 
        #print(numeros) # Coloca no array os elementos uma só vez (evita repetições)
        aux = max(numeros) 
            
        numero_ocorrencias = np.bincount(n_array) #Conta o numero de ocorrencias de cada numero
        numero_ocorrencias[0]=0 
        #print(numero_ocorrencias)
            
        #Passo 2 (DOC 2)
        prox_codigo = np.zeros(aux+1,int)
        codigo = 0
            
        for bits in range(1,aux+1):
            codigo = (codigo +  numero_ocorrencias[bits-1]) << 1 # Por ex: codigo 5 começa no numero 28, 4 no numero 12,3 no numero zero  
            prox_codigo[bits] = codigo
        #print(prox_codigo)
            
        #Passo 3 (DOC 2)
            
        codigo_arvore = []
        for i in range(max(ordem)+1):
            codigo_arvore.append("") #Tornar os elementos como strings
                
        for n in range(len(n_array)):
              comp = n_array[n]
              if comp != 0:
                  codigo_arvore[n] = bin(prox_codigo[comp])[2:]
                  #print(bin(prox_codigo[comp])[2:])
                  prox_codigo[comp] +=1
              if n_array[n] != len(codigo_arvore[n]):
                 codigo_arvore[n] = '0'*(n_array[n]-len(codigo_arvore[n])) + codigo_arvore[n]
                  
        return codigo_arvore
    
    
    def inserirTree(self,codigo_arvore):
        HuffmanCode = HuffmanTree()
        for i in range (len(codigo_arvore)) :
            if codigo_arvore[i] != '':
                HuffmanCode.addNode(codigo_arvore[i],i,True)
        return HuffmanCode
  
    def exercicio4_5(self,alfabeto,codigo_arvore):
         code = []
         count = 0
         read_Bits1 = ""
            
         while alfabeto != count:
             read_Bits1 = read_Bits1 + str(self.readBits(1))
             if read_Bits1 in codigo_arvore:
                 pos = codigo_arvore.index(read_Bits1)
                 
                 if pos == 16:
                     num_vezes = self.readBits(2)
                     comp_anterior = code[len(code)-1]
                     for i in range(num_vezes+3):
                         code.append(comp_anterior)
                         count = count +1
                 elif pos == 17:
                    num_vezes = self.readBits(3)
                    for i in range(num_vezes+3):
                        code.append(0)
                        count = count +1
                        
                 elif pos == 18:
                    num_vezes = self.readBits(7)
                    for i in range(num_vezes + 11):
                        code.append(0)
                        count = count +1
                 else:
                    code.append(pos)
                    count = count+1
                        
                 read_Bits1 = ""
        
         return code
     
    def exercicio6(self,code1,code):
        
      ordem = []
      for j in range (len(code)):
          ordem.append(j)
          
      novo_array = self.exercicio3(code, ordem)    
      
      treeCodeLiterais = self.inserirTree(novo_array) 
        
      ordem1 = []
      for i in range (len(code1)):
          ordem1.append(i)
           
      novo_array1 = self.exercicio3(code1, ordem1)

      
      treeCodeDist = self.inserirTree(novo_array1)
      
      return novo_array, novo_array1, treeCodeLiterais, treeCodeDist
  
    def exercicio7(self,treeCodeLiterais,treeCodeDist):
        ler_literal = ""
        ler_Distancia = ""
        texto = ""
        vetor = []
        code_literal = treeCodeLiterais.findNode(ler_literal)
        code_dist = treeCodeDist.findNode(ler_Distancia)
        
        
        while (code_literal != 256):
            while (code_literal == -2):
                ler_literal += bin(self.readBits(1))[2:]
                code_literal = treeCodeLiterais.findNode(ler_literal)
                
            if (code_literal < 256):
                vetor.append(code_literal)
                texto += chr(code_literal)
                
            
            elif (code_literal == 256):
                break
            else:
                if (257 <= code_literal <= 264):
                    comp = (code_literal - 257) + 3
                    
                elif (265 <= code_literal <= 268):
                    comp = 2*(code_literal - 265) + 11 + self.readBits(1)
                elif (269 <= code_literal <= 272):
                    comp = 4*(code_literal - 269) + 19 + self.readBits(2)
                elif (273 <= code_literal <= 276):
                    comp = 8*(code_literal - 273) + 35 + self.readBits(3)
            
                        
                while (code_dist == -2):
                    ler_Distancia += bin(self.readBits(1))[2:]
                    code_dist = treeCodeDist.findNode(ler_Distancia)
                    
                if (code_dist <= 3):    
                    recuar = code_dist + 1
                elif(code_dist == 4):
                     recuar = 5 + self.readBits(1)
                elif(code_dist == 5):
                    recuar = 7 + self.readBits(1)
                elif(code_dist == 6):
                        recuar = 9 + self.readBits(2)
                elif(code_dist == 7):
                        recuar = 13 + self.readBits(2) 
                elif(code_dist == 8):
                        recuar = 17 + self.readBits(3)
                elif(code_dist == 9):
                        recuar = 25 + self.readBits(3)
                elif(code_dist == 10):
                        recuar = 33 + self.readBits(4)
                elif(code_dist == 11):
                        recuar = 49 + self.readBits(4)
                elif(code_dist == 12):
                        recuar = 65 + self.readBits(5)
                elif(code_dist == 13):
                        recuar = 97 + self.readBits(5)
                elif(code_dist == 14):
                        recuar = 129 + self.readBits(6)
                elif(code_dist == 15):
                        recuar = 193 + self.readBits(6)
                elif(code_dist == 16):
                        recuar = 257 + self.readBits(7)
                elif(code_dist == 17):
                        recuar = 385 + self.readBits(7)
                elif(code_dist == 18):
                        recuar = 513 + self.readBits(8)
                elif(code_dist == 19):
                        recuar = 769 + self.readBits(8)
                elif(code_dist == 20):
                        recuar = 1025 + self.readBits(9)
                elif(code_dist == 21):
                        recuar = 1537 + self.readBits(9)
                elif(code_dist == 22):
                        recuar = 2049 + self.readBits(10)
                elif(code_dist == 23):
                        recuar = 3073 + self.readBits(10)
        
                for i in range(comp):
                            vetor.append(vetor[-int(recuar)])
                            texto += chr(vetor[-1])
                            
            ler_literal = ""
            ler_Distancia = ""
                
            code_literal = treeCodeLiterais.findNode(ler_literal)
            code_dist = treeCodeDist.findNode(ler_Distancia)
         
        return vetor, texto
    

if __name__ == '__main__':

    # gets filename from command line if provided
    filename = "FAQ.txt.gz"
    if len(sys.argv) > 1:
        filename = sys.argv[1]

    # decompress file
    
    GZIPHeader.fName = "FAQ.txt"
    gz = GZIP(filename)
    gz.decompress()
