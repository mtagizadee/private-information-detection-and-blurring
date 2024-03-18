import random
import nltk
import time
import faker
import pandas as pd
import numpy as np

nltk.download('words')
from nltk.corpus import words

fake = faker.Faker()

def main():
  n, m = 1000, 100
  out = "./data/pii_labeled.csv"

  non_pii = seed_non_pii(n)
  pii = seed_pii(m)

  take_non_pii = 100
  take_pii = 50

  non_pii = non_pii[:take_non_pii]
  pii = pii[:take_pii]

  total = non_pii + pii
  labels = [0] * len(non_pii) + [1] * len(pii)

  data = {
    "text": total,
    "label": labels
  }

  df = pd.DataFrame(data).to_numpy()
  np.random.shuffle(df)

  df = pd.DataFrame(df, columns=["text", "label"])
  
  df.to_csv(out, index=False)

def seed_non_pii(n):
  word_list = words.words()

  random.seed(time.time())
  random.shuffle(word_list)

  selected_words = word_list[:n]

  return selected_words

def seed_pii(n):
  pii = []

  for _ in range(n):
    data = {
        "email": custom_mail(),
        "phone": fake.phone_number(),
        "address": fake.address(),
        "name": fake.name(),
        "credit_card": fake.credit_card_number(card_type=None),
        "driver_license": fake.license_plate(),  # Simulating as license plate due to Faker's limitations
        "ssn": fake.ssn(),
        "medical_record": f"MR{fake.random_number(digits=6, fix_len=True)}",
        "health_insurance": f"HI{fake.random_number(digits=9, fix_len=True)}",
        "bank_account": fake.bban(),
        "vin": fake.bothify(text='?#??###?#?######', letters='ABCDEFGHJKLMNPRSTUVWXYZ'),  # Custom VIN-like pattern
        "license_plate": fake.license_plate()
    }

    pii.append(data["email"])
    pii.append(data["phone"])
    pii.append(data["address"])
    pii.append(data["name"])
    pii.append(data["credit_card"])
    pii.append(data["driver_license"])
    pii.append(data["ssn"])
    pii.append(data["medical_record"])
    pii.append(data["health_insurance"])
    pii.append(data["bank_account"])
    pii.append(data["vin"])
    pii.append(data["license_plate"])
    

  return pii

def custom_mail():
  username = fake.user_name()
  domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com"]

  return f"{username}@{random.choice(domains)}"

if __name__ == "__main__":
  main()