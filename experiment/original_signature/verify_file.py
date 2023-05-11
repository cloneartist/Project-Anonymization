import csv
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.hazmat.primitives import serialization
# Verify the signature using the public key
file_path = 'experiment/data.csv'
public_key_path = 'public.pem'
with open(file_path, 'rb') as file:
    file_data = file.read()
    hasher = hashes.Hash(hashes.SHA256())
    hasher.update(file_data)
    digest = hasher.finalize()
    with open('sign.txt', 'rb') as signature_file:
        signature = signature_file.read()
        with open(public_key_path, 'rb') as public_key_file:
            public_key = load_pem_public_key(public_key_file.read())
            try:
                public_key.verify(signature, digest, padding.PKCS1v15(), hashes.SHA256())
                print("File has not been tampered with.")
            except:
                print("File has been tampered with.")