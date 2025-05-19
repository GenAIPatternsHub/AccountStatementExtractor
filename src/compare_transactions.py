import json

# Charger les données JSON
with open('./output/step2.json') as f:
    data = json.load(f)

# Initialiser les montants
initial_amount = data['initial_amount']
final_amount = data['final_amount']

# Calculer le montant à partir des transactions
calculated_amount = initial_amount + sum(transaction['amount'] for transaction in data['transactions'])

# Comparer les montants
if calculated_amount == final_amount:
    print("Le montant final est correct.")
    print(f"calculé : {calculated_amount} == observé : {final_amount}")
else:
    print(f"Erreur: le montant calculé est {calculated_amount}, mais le montant final est {final_amount}.")
    
    
    
    
    