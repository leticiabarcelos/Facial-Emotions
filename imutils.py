# Imports
import numpy as np
import cv2
import pandas as pd

def translate(image, x, y):
	# Define a matriz de tradução e realiza a tradução
	M = np.float32([[1, 0, x], [0, 1, y]])
	shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

	# Retorna a imagem traduzida
	return shifted

def rotate(image, angle, center = None, scale = 1.0):
	# Obtém as dimensões da imagem
	(h, w) = image.shape[:2]

	# Se o centro for None, inicialize-o como o centro da imagem
	if center is None:
		center = (w / 2, h / 2)

	# Executa a rotação
	M = cv2.getRotationMatrix2D(center, angle, scale)
	rotated = cv2.warpAffine(image, M, (w, h))

	# Retorna a imagem girada
	return rotated

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# Inicializa as dimensões da imagem a ser redimensionada e pegue o tamanho da imagem
	dim = None
	(h, w) = image.shape[:2]

	# Se tanto a largura quanto a altura são None, então retorna a imagem original
	if width is None and height is None:
		return image

	# Verifica se a largura é None
	if width is None:
		# Calcula a proporção da altura e constrói as dimensões
		r = height / float(h)
		dim = (int(w * r), height)

	# Caso contrário, a altura é None
	else:
		# Calcula a proporção da largura e constrói as dimensões
		r = width / float(w)
		dim = (width, int(h * r))

	# Redimensiona a imagem
	resized = cv2.resize(image, dim, interpolation = inter)

	# Devolve a imagem redimensionada
	return resized

def exibirPorcentagemEmocoes(frame, countEmocoes):
    x = 50
    y = 370

    x2 = 50
    y2 = 400

    x3 = 50
    y3 = 430

    p1, p2, p3 = getPorcentagens(countEmocoes)
    print('probabilidades')
    print(p1)
    print(p2)
    if list(p1.keys())[0] == 3:
        cv2.putText(frame, str(round(list(p1.values())[0],3))+"% Feliz",(x,y), cv2.FONT_ITALIC, 1, 255, 3)
    elif list(p1.keys())[0] == 0:
        cv2.putText(frame, str(round(list(p1.values())[0],3))+"% Nervoso",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3)
    elif list(p1.keys())[0] == 1:
        cv2.putText(frame, str(round(list(p1.values())[0],3))+"% Esnobe",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3)
    elif list(p1.keys())[0] == 2:
        cv2.putText(frame, str(round(list(p1.values())[0],3))+"% Com Medo",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3)
    elif list(p1.keys())[0] == 4:
        cv2.putText(frame, str(round(list(p1.values())[0],3))+"% Triste",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3)
    elif list(p1.keys())[0] == 5:
        cv2.putText(frame, str(round(list(p1.values())[0],3))+"% Surpreso",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3)
    else :
        cv2.putText(frame, str(round(list(p1.values())[0],3))+"% Neutro",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3)

    if list(p2.keys())[0] == 3:
        cv2.putText(frame, str(round(list(p2.values())[0],3))+"% Feliz",(x2,y2), cv2.FONT_ITALIC, 1, 255, 3)
    elif list(p2.keys())[0] == 0:
        cv2.putText(frame, str(round(list(p2.values())[0],3))+"% Nervoso",(x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 1, 155, 3)
    elif list(p2.keys())[0] == 1:
        cv2.putText(frame, str(round(list(p2.values())[0],3))+"% Esnobe",(x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 1, 155, 3)
    elif list(p2.keys())[0] == 2:
        cv2.putText(frame, str(round(list(p2.values())[0],3))+"% Com Medo",(x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 1, 155, 3)
    elif list(p2.keys())[0] == 4:
        cv2.putText(frame, str(round(list(p2.values())[0],3))+"% Triste",(x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 1, 155, 3)
    elif list(p2.keys())[0] == 5:
        cv2.putText(frame, str(round(list(p2.values())[0],3))+"% Surpreso",(x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 1, 155, 3)
    else :
        cv2.putText(frame, str(round(list(p2.values())[0],3))+"% Neutro",(x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 1, 155, 3)

    if list(p3.keys())[0] == 3:
        cv2.putText(frame, str(round(list(p3.values())[0],3))+"% Feliz",(x3,y3), cv2.FONT_ITALIC, 1, 255, 3)
    elif list(p3.keys())[0] == 0:
        cv2.putText(frame, str(round(list(p3.values())[0],3))+"% Nervoso",(x3,y3), cv2.FONT_HERSHEY_SIMPLEX, 1, 100, 3)
    elif list(p3.keys())[0] == 1:
        cv2.putText(frame, str(round(list(p3.values())[0],3))+"% Esnobe",(x3,y3), cv2.FONT_HERSHEY_SIMPLEX, 1, 100, 3)
    elif list(p3.keys())[0] == 2:
        cv2.putText(frame, str(round(list(p3.values())[0],3))+"% Com Medo",(x3,y3), cv2.FONT_HERSHEY_SIMPLEX, 1, 100, 3)
    elif list(p3.keys())[0] == 4:
        cv2.putText(frame, str(round(list(p3.values())[0],3))+"% Triste",(x3,y3), cv2.FONT_HERSHEY_SIMPLEX, 1, 100, 3)
    elif list(p3.keys())[0] == 5:
        cv2.putText(frame, str(round(list(p3.values())[0],3))+"% Surpreso",(x3,y3), cv2.FONT_HERSHEY_SIMPLEX, 1, 100, 3)
    else :
        cv2.putText(frame, str(round(list(p3.values())[0],3))+"% Neutro",(x3,y3), cv2.FONT_HERSHEY_SIMPLEX, 1, 100, 3)

def getPorcentagens(countEmocoes):
	total = 0
	for i in countEmocoes:
		total += countEmocoes[i]
	result = {}
	for i in countEmocoes:
		result[i] = calculaPorcentagem(countEmocoes[i], total)

	index = []
	value = []
	for i in range(6):
		index.append(i)
		value.append(result[i])
	dicionario = {}
	dicionario['index'] = index
	dicionario['value'] = value
	df = pd.DataFrame(dicionario)
	df = df.sort_values(['value'], ascending=False)
	print(result)
	print(df)
	p1 = {}

	p2 = {}

	p3 = {}

	count = 0
	df = df.iloc[:,:].values
	print(df)
	for data in df:
		print(data)
		if count > 2:
			break
		if count == 0:
			p1[data[0]] = data[1]
		elif count == 1:
			p2[data[0]] = data[1]
		elif count == 2:
			p3[data[0]] = data[1]
		count += 1
	return p1, p2, p3



def calculaPorcentagem(quantidade, total):
	return quantidade / total * 100