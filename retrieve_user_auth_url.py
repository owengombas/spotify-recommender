from lib.spoti import SpotiUser

ID = """
tea
"""


user = SpotiUser(ID.strip())
print(user.get_auth_url())

print()
user.log_user()
print(user.get_email())
