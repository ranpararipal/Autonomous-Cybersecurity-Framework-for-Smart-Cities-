import paho.mqtt.client as mqtt
import random
import time
import threading

# Agent class
class Agent:
    def __init__(self, name, broker_address="localhost"):
        self.name = name
        self.broker_address = broker_address
        self.client = mqtt.Client(name)
        self.client.connect(broker_address)
        self.client.on_message = self.on_message

    def on_message(self, client, userdata, message):
        print(f"{self.name} received message: {message.payload.decode()}")
    
    def send_message(self, topic, message):
        self.client.publish(topic, message)

    def subscribe(self, topic):
        self.client.subscribe(topic)
    
    def start(self):
        self.client.loop_start()


# Function to create and run agents
def run_agents():
    # Creating two agents for example
    agent1 = Agent("Agent1")
    agent2 = Agent("Agent2")
    
    agent1.subscribe("smart_city/alerts")
    agent2.subscribe("smart_city/alerts")
    
    agent1.start()
    agent2.start()

    # Simulate agents sending messages
    while True:
        agent1.send_message("smart_city/alerts", "Threat detected!")
        time.sleep(5)
        agent2.send_message("smart_city/alerts", "Mitigation response sent.")
        time.sleep(5)

# Start agents in a separate thread
agent_thread = threading.Thread(target=run_agents)
agent_thread.start()
