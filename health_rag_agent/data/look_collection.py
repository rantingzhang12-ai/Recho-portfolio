from pymilvus import connections, utility

from pymilvus import Collection


connections.connect(host="localhost", port="19530")

collection = Collection("health_rag")


print(utility.list_collections())

print(collection.num_entities)

results = collection.query(
    expr="",
    output_fields=["*"],
    limit=5
)

for r in results:
    print(r)
