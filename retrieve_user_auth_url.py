"""
Use this script to retrieve the auth url for a user
1) Ask the person email
2) Add the email to the authorized users list (https://developer.spotify.com/dashboard/)
3) User logs in and authorizes the app
4) The person is redirected to the redirect url with the auth code
5) The authcode should be send back to us to complete the auth process
6) Once the auth process is complete, we can retrieve the user's data
"""

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
