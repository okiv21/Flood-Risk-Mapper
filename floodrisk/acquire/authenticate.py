"""Authenticate locally with verified CA certificates and initialise Earth Engine."""
import argparse
import os


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',required=True)
    args=parser.parse_args()
    import certifi
    # Python.org macOS installations may not have a usable default CA bundle.
    # Preserve TLS verification using the installed public certificate bundle.
    os.environ['SSL_CERT_FILE']=certifi.where()
    import ee
    ee.Authenticate(auth_mode='localhost',scopes=[
        'https://www.googleapis.com/auth/earthengine',
        'https://www.googleapis.com/auth/cloud-platform'])
    ee.Initialize(project=args.project)
    print(f'Earth Engine initialized successfully with {args.project}.')


if __name__=='__main__':main()
