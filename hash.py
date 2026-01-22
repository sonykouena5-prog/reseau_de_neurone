import streamlit_authenticator as stauth

hashed = stauth.Hasher(["sony@1342"]).generate()
print(hashed[0])
