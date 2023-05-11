import csv
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization

# Set the file paths
file_path = 'experiment\data.csv'
public_key_path = 'public.pem'

# Generate a private/public key pair
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)
public_key = private_key.public_key()

# Save the public key to a file
with open(public_key_path, 'wb') as public_key_file:
    public_key_file.write(public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ))

# Sign the CSV file
with open(file_path, 'rb') as file:
    file_data = file.read()
    hasher = hashes.Hash(hashes.SHA256())
    hasher.update(file_data)
    digest = hasher.finalize()
    signature = private_key.sign(digest, padding.PKCS1v15(), hashes.SHA256())

# Write the signature to a separate file
with open('sign.txt', 'wb') as signature_file:
    signature_file.write(signature)
    print("File Signed By Authors")