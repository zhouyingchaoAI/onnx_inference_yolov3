import requests
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--file", type=str, help="the pic path")
args = parser.parse_args()

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open(args.file,'rb')})

print(resp.text)