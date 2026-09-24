import json

with open("reply_networks/politics.json", "r") as fp:
    month_nets = json.load(fp)

print("Number of monthly networks:", len(month_nets))
print("Type of first month:", type(month_nets[0]))

print("\nFirst 5 users:")
for user, replies in list(month_nets[0].items())[:5]:
    print(user, "->", replies)