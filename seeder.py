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

  take_non_pii = 800
  take_pii = 300

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
    card = {
      "card_number": fake.credit_card_number(card_type=None),
      "security_code": "{:03d}".format(random.randint(0, 999)), 
      "expiration_date": fake.credit_card_expire(start="now", end="+10y", date_format="%m/%y"),
      "card_holder_name": fake.name(),
    }

    personal = {
      "address": fake.address(),
      "email": fake.email(),
      "phone_number": fake.phone_number(),
      "personal_id": fake.ssn(), 
      "passport_number": fake.bothify(text="????######", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
      "date_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=90).isoformat(),
      "nationality": fake.country(),
      "date": fake.future_date(end_date="+10y", tzinfo=None).isoformat(), 
    }

    pii.append(personal["full_name"])
    pii.append(personal["address"])
    pii.append(personal["email"])
    pii.append(personal["phone_number"])
    pii.append(personal["personal_id"])
    pii.append(personal["passport_number"])
    pii.append(personal["date_of_birth"])
    pii.append(personal["nationality"])
    pii.append(personal["date"])

    pii.append(card["card_number"])
    pii.append(card["security_code"])
    pii.append(card["expiration_date"])
    pii.append(card["card_holder_name"])

  return pii

if __name__ == "__main__":
  main()