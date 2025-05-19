import pypdf
import pytesseract
from PIL import Image
import io
import os

# Assurez-vous que Tesseract est installé sur votre système
# Sous Ubuntu, vous pouvez l'installer avec : sudo apt install tesseract-ocr
# Sous macOS, vous pouvez l'installer avec : brew install tesseract
# Sous Windows, vous pouvez le télécharger depuis le site officiel de Tesseract

# Chemin vers le fichier PDF
pdf_path = './input/releve_compte_12_2024_012.pdf'

# Chemin vers le fichier de sortie
output_path = './output/step1.txt'

# Créer le répertoire de sortie s'il n'existe pas
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Ouvrir le fichier PDF
with open(pdf_path, 'rb') as file:
    pdf_reader = pypdf.PdfReader(file)

    # Parcourir chaque page du PDF
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]

        # Extraire le texte de la page
        text = page.extract_text()

        # Si le texte extrait est vide, essayer d'effectuer l'OCR
        if not text:
            # Convertir la page en image
            page_image = page.to_image()

            # Sauvegarder l'image temporairement
            image_path = f'temp_page_{page_num}.png'
            page_image.save(image_path)

            # Effectuer l'OCR sur l'image
            text = pytesseract.image_to_string(Image.open(image_path))

            # Supprimer l'image temporaire
            os.remove(image_path)

        # Écrire le texte extrait dans le fichier de sortie
        with open(output_path, 'a', encoding='utf-8') as output_file:
            output_file.write(f'Page {page_num + 1}:\n')
            output_file.write(text)
            output_file.write('\n\n')

print(f'Le texte a été extrait et sauvegardé dans {output_path}')

