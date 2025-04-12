import json
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Path to the JSON file
users_file_path = Path("c:/Users/User/Desktop/thriveai-ami-ui/backend/users.json")


# Function to load users from JSON file
def load_users():
    if users_file_path.exists():
        with open(users_file_path, "r") as file:
            return json.load(file)
    return {}


# Function to save users to JSON file
def save_users(users):
    # Ensure each user is converted to a dictionary
    users_dict = {username: user if isinstance(user, dict) else user.dict() for username, user in users.items()}
    with open(users_file_path, "w") as file:
        json.dump(users_dict, file)


# Load users from JSON file
users_db = load_users()


# User model
class User(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = None


# User registration model
class UserInDB(User):
    hashed_password: str


# User registration input model
class UserCreate(BaseModel):
    username: str
    email: str
    password: str


# Function to hash passwords
def hash_password(password: str):
    return pwd_context.hash(password)


# Function to verify passwords
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


# Endpoint to register a new user
@app.post("/register")
def register(user: UserCreate):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = hash_password(user.password)
    users_db[user.username] = UserInDB(**user.dict(), hashed_password=hashed_password)
    save_users(users_db)
    return {"msg": "User registered successfully"}


# Endpoint to login and get a token
@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    users_db = load_users()

    # Find user by email instead of username
    user = None
    for u in users_db.values():
        if u['email'] == form_data.username:  # form_data.username will contain email
            user = u
            break

    if not user or not verify_password(form_data.password, user['hashed_password']):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    return {"access_token": user['username'], "token_type": "bearer"}


# Endpoint to get user details
@app.get("/users/me")
def read_users_me(token: str = Depends(oauth2_scheme)):
    user = users_db.get(token)
    if user is None:
        raise HTTPException(status_code=400, detail="Invalid token")
    return user
