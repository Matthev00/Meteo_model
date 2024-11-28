import dotenv
import os


def load_env():
    dotenv_path = os.path.join("./", '.env')
    dotenv.load_dotenv(dotenv_path)


if __name__ == "__main__":
    load_env()
    
