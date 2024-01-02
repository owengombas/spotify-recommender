from lib.spoti import SpotiUser
from faker import Faker

faker = Faker()
anonimized = faker.name().split(" ")[0].lower()

ID = f"""
{anonimized}
"""


user = SpotiUser(ID.strip())
print(user.get_auth_url())

print()
user.log_user()
print(user.get_email())
