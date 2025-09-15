import cv2
import numpy as np
import time


def iterative_threshold(magnitude_img):
    """
    Implementa o filtro de limiar iterativo descrito na Seção III-D do artigo.
    Isso calcula o limiar T ideal de forma iterativa[cite: 159].
    """
    # 1) Obter limiar inicial T (média do gradiente máx e mín) [cite: 160, 161, 162]
    T0 = np.min(magnitude_img)
    T1 = np.max(magnitude_img)
    T = (T0 + T1) / 2.0

    # Define uma tolerância pequena para a convergência
    allowance = 1.0  # Tolerância em valor de gradiente

    while True:
        # 2) Calcular Ta (média dos gradientes <= T) e Tb (média dos gradientes > T) [cite: 163, 164]
        # Filtrando valores zero para evitar distorção na média (assumindo que 0 não é uma borda)
        mag_gt_T = magnitude_img[magnitude_img > T]
        mag_lte_T = magnitude_img[
            (magnitude_img <= T) & (magnitude_img > 0)
        ]  # Ignora fundo puro

        if mag_gt_T.size == 0:
            Tb = 0
        else:
            Tb = np.mean(mag_gt_T)

        if mag_lte_T.size == 0:
            Ta = 0
        else:
            Ta = np.mean(mag_lte_T)

        # 3) Calcular novo limiar TT [cite: 172, 174]
        TT = (Ta + Tb) / 2.0

        # 4) Verificar convergência [cite: 181, 182]
        if abs(T - TT) < allowance:
            break

        T = TT

    return T


def non_maximum_suppression(magnitude, angle):
    """
    Implementação manual da Supressão Não Máxima (NMS).
    [Ref: Etapa do fluxograma em Fig. 2 (fonte: 101)]
    """
    (H, W) = magnitude.shape
    nms_output = np.zeros((H, W), dtype=np.float32)

    # Converte ângulos de radianos para graus
    degrees = angle * 180.0 / np.pi
    degrees[degrees < 0] += 180

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            q = 255
            r = 255

            # Quantiza o ângulo em 4 direções (0, 45, 90, 135)
            # 0 graus (horizontal)
            if (0 <= degrees[i, j] < 22.5) or (157.5 <= degrees[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            # 45 graus
            elif 22.5 <= degrees[i, j] < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            # 90 graus (vertical)
            elif 67.5 <= degrees[i, j] < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            # 135 graus
            elif 112.5 <= degrees[i, j] < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            # Se o pixel atual (i, j) for o máximo local ao longo da direção do gradiente, mantenha-o
            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                nms_output[i, j] = magnitude[i, j]
            else:
                nms_output[i, j] = 0

    return nms_output


def improved_canny_algorithm(image_path, alpha=0.75):
    """
    Função principal que implementa o fluxograma da Fig. 2 [cite: 96-103].

    Args:
        image_path (str): Caminho para a imagem de entrada.
        alpha (float): Coeficiente para Resolução de Gradiente (deve estar em [0.5, 1.0])[cite: 186].
    """

    # 1. Imagem de Entrada
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro: Não foi possível carregar a imagem em {image_path}")
        return None, None

    # 2. Gray Scaling [cite: 98]
    # O artigo usa uma fórmula de média ponderada específica[cite: 93],
    # que é o padrão usado pelo OpenCV (BT.601 / BT.709)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Median Filter [cite: 99]
    # O artigo substitui o filtro Gaussiano pelo Mediano[cite: 29].
    # O artigo indica que um kernel 3x3 teve bom desempenho[cite: 130].
    median_img = cv2.medianBlur(gray_img, 3)

    # 4. Cálculo de Gradiente (Operador Sobel Melhorado) [cite: 137]
    # Usando os kernels exatos definidos nas equações (2) e (3) do artigo

    sqrt2_4 = np.sqrt(2) / 4.0

    # Kernel Gx [cite: 139]
    kernel_x = np.array(
        [[-sqrt2_4, 0, sqrt2_4], [-1, 0, 1], [-sqrt2_4, 0, sqrt2_4]], dtype=np.float32
    )

    # Kernel Gy [cite: 142]
    kernel_y = np.array(
        [[-sqrt2_4, -1, -sqrt2_4], [0, 0, 0], [sqrt2_4, 1, sqrt2_4]], dtype=np.float32
    )

    # Aplicar os filtros. Usamos CV_64F para precisão e para capturar valores negativos.
    Gx = cv2.filter2D(median_img, cv2.CV_64F, kernel_x)
    Gy = cv2.filter2D(median_img, cv2.CV_64F, kernel_y)

    # Calcular Magnitude e Ângulo (Direção) do Gradiente
    magnitude = np.sqrt(Gx**2 + Gy**2)
    angle = np.arctan2(Gy, Gx)  # arctan2 lida com todos os quadrantes

    # 5. Iterative gradient threshold filter (Filtro Iterativo) [cite: 100]
    # Isso calcula o Limiar T (T_converged) [cite: 159-183]
    print("Iniciando cálculo do limiar iterativo...")
    T_converged = iterative_threshold(magnitude)
    print(f"Limiar T iterativo convergido: {T_converged:.4f}")

    # 6. Gradient Resolution (Resolução de Gradiente)
    # Calcula K = alpha * T^2 e zera todos os gradientes abaixo de K
    K = alpha * (T_converged**2)
    print(f"Calculando K (Resolução) com alpha={alpha}: K = {K:.4f}")

    # Criar o mapa de magnitude filtrado: qualquer valor < K é definido como 0
    filtered_magnitude = magnitude.copy()
    filtered_magnitude[filtered_magnitude < K] = 0

    # 7. Non Maximum Suppression (NMS) [cite: 101]
    # Aplicamos NMS ao mapa de magnitude JÁ FILTRADO (usando os ângulos originais)
    print("Aplicando Supressão Não Máxima (NMS)...")
    nms_img = non_maximum_suppression(filtered_magnitude, angle)

    # 8. Single threshold setting and binarization
    # Como o NMS zera todos os pixels que não são picos de borda,
    # e já zeramos todos os picos fracos (abaixo de K),
    # qualquer valor não-zero restante no mapa NMS é uma borda forte.
    # A binarização é simplesmente definir esses pixels restantes para 255.

    # Normalizar para 8 bits (0-255)
    if np.max(nms_img) > 0:
        nms_img_norm = (nms_img / np.max(nms_img)) * 255
    else:
        nms_img_norm = nms_img

    final_output = nms_img_norm.astype(np.uint8)

    # Uma binarização final simples para garantir uma saída puramente P&B.
    # Qualquer pixel que sobreviveu ao NMS é uma borda.
    _, final_binary = cv2.threshold(final_output, 1, 255, cv2.THRESH_BINARY)

    print("Processamento concluído.")
    return (
        final_binary,
        nms_img,
    )  # Retorna a imagem binária e o mapa de borda NMS (para depuração)


# --- EXEMPLO DE USO ---

# Substitua 'sua_imagem.jpg' pelo caminho da sua imagem de teste
# (Usei a imagem (a) da Fig. 5 como referência conceitual)
IMAGE_FILE = "ship_image.png"  # Você deve fornecer esta imagem

# Gerar uma imagem de navio de exemplo se ela não existir (apenas para teste)
try:
    cv2.imread(IMAGE_FILE).shape
except AttributeError:
    print("Gerando imagem de teste 'ship_image.png'...")
    ship_img_mock = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(ship_img_mock, (100, 200), (540, 350), (200, 200, 200), -1)
    cv2.line(ship_img_mock, (100, 200), (200, 150), (220, 220, 220), 20)
    cv2.line(ship_img_mock, (400, 150), (540, 200), (220, 220, 220), 10)
    # Adicionar algum ruído "sal e pimenta" para testar o filtro mediano
    noise = np.random.randint(0, 100, (480, 640, 3))
    ship_img_mock[noise < 5] = 255  # Pimenta
    ship_img_mock[noise > 95] = 0  # Sal
    cv2.imwrite(IMAGE_FILE, ship_img_mock)


start_time = time.time()
improved_canny_output, _ = improved_canny_algorithm(IMAGE_FILE, alpha=0.7)

print(improved_canny_output.shape)

end_time = time.time()

print(f"\nTempo de execução do Canny Melhorado: {end_time - start_time:.4f} segundos")

# Comparação com o Canny padrão do OpenCV (para referência)
original_img = cv2.imread(IMAGE_FILE)
gray_orig = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
# O Canny padrão usa filtro Gaussiano e Histerese (limiar duplo)
opencv_canny = cv2.Canny(gray_orig, 50, 150)


if improved_canny_output is not None:
    cv2.imshow("Original com Ruido", original_img)
    cv2.imshow("Canny Padrao (OpenCV)", opencv_canny)
    cv2.imshow("Canny Melhorado (Artigo)", improved_canny_output)

    print(
        "\nResultados sendo exibidos. Pressione qualquer tecla para fechar as janelas."
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

