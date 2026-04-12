from chromadb import PersistentClient

client = PersistentClient(path="C:\\Users\\An Tran\\Desktop\\physbot\\data\\chroma_db")
collection = client.get_collection("physbot_sgk")

print("Số chunks trong DB:", collection.count())