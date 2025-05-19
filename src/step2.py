from dotenv import load_dotenv
# Chargement des variables d'environnement à partir du fichier .env
load_dotenv()

from typing import cast
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
import json
from pydantic import BaseModel, Field
import os



# Fonction pour convertir un fichier PDF en texte
def convert_pdf_as_text(path: str) -> Document:
    loader = PyPDFLoader(path, mode="single")  # Charger le PDF en mode "single"
    documents = loader.load()  # Charger le contenu du PDF
    return documents[0]  # Retourner le premier document

# Extraction du contenu du PDF converti en texte
content = convert_pdf_as_text('./input/releve_compte_12_2024_012.pdf').page_content

# Création d'un modèle de prompt pour l'analyse des transactions bancaires
prompt = PromptTemplate.from_template("""
Vous êtes un assistant spécialisé dans l'extraction de transactions financières à partir de relevés de comptes. 
Votre tâche est d'identifier et d'extraire les informations pertinentes des transactions, telles que la date, le montant et la description de la transaction.

Instructions :
1. Identifiez chaque transaction dans le relevé de comptes.
2. Pour chaque transaction, extrayez les informations suivantes :
   - Date de la transaction
   - Montant de la transaction
   - Description de la transaction
3. Organisez les informations extraites dans un format structuré

Le relevé de compte à analyser :
{data}

""")

# Classe représentant une transaction dans un relevé de compte
class Transaction(BaseModel):
    """
    Modélise une transaction dans un relevé de compte.
    Attributes:
        date (str): Date de la transaction.
        label (str): Description ou identifiant de la transaction.
        amount (float): Montant de la transaction.
    """
    date: str = Field(description="Date de la transaction")  # Date de la transaction
    label: str = Field(description="Label de la transaction")  # Label de la transaction
    amount: float = Field(description="Montant de la transaction")  # Montant de la transaction

# Classe représentant un relevé de compte contenant une liste de transactions
class AccountStatement(BaseModel):
    """
    Modélise un relevé de compte contenant des informations financières.

    Attributes:
        initial_amount (float): Montant initial sur le relevé de compte.
        final_amount (float): Montant final sur le relevé de compte après toutes les transactions.
        transactions (list[Transaction]): Liste des transactions associées à ce relevé de compte.
    """
    initial_amount: float = Field(description="Montant initial")
    final_amount: float = Field(description="Montant final")
    transactions: list[Transaction] = Field(description="Liste des transactions")  # Liste des transactions
    
    
# Initialisation du modèle de langage avec ChatOpenAI en choisissant le modèle "gpt-4o-mini"
llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(AccountStatement)

# Exécution du prompt avec le modèle de langage et obtention d'un résultat structuré
res = cast("AccountStatement", (prompt | llm).invoke({
    "data": content  # Passer le contenu comme entrée
}))

# Enregistrement du résultat au format JSON dans le répertoire output
output_path = './output/step2.json'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as json_file:
    json.dump(res.model_dump(), json_file, indent=3, ensure_ascii=False)

# Traitement de tous les fichiers PDF dans le répertoire "./input"
input_directory = './input'
output_directory = './output'

for filename in os.listdir(input_directory):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(input_directory, filename)

        # Extraction du contenu du PDF converti en texte
        content = convert_pdf_as_text(pdf_path).page_content

        # Exécution du prompt avec le modèle de langage et obtention d'un résultat structuré
        res = cast("AccountStatement", (prompt | llm).invoke({
            "data": content  # Passer le contenu comme entrée
        }))

        # Détermination du chemin de sortie avec le même nom que le PDF mais avec l'extension .json
        json_filename = f"{os.path.splitext(filename)[0]}.json"
        output_path = os.path.join(output_directory, json_filename)

        # Enregistrement du résultat au format JSON dans le répertoire output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(res.model_dump(), json_file, indent=3, ensure_ascii=False)# Traitement de tous les fichiers PDF dans le répertoire "./input"
input_directory = './input'
output_directory = './output'

for filename in os.listdir(input_directory):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(input_directory, filename)

        # Extraction du contenu du PDF converti en texte
        content = convert_pdf_as_text(pdf_path).page_content

        # Exécution du prompt avec le modèle de langage et obtention d'un résultat structuré
        res = cast("AccountStatement", (prompt | llm).invoke({
            "data": content  # Passer le contenu comme entrée
        }))

        # Détermination du chemin de sortie avec le même nom que le PDF mais avec l'extension .json
        json_filename = f"{os.path.splitext(filename)[0]}.json"
        output_path = os.path.join(output_directory, json_filename)

        # Enregistrement du résultat au format JSON dans le répertoire output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(res.model_dump(), json_file, indent=3, ensure_ascii=False)