from mas_model import Agent
import paho.mqtt.client as mqtt

# Create and run agents
def run_agents():
    agent1 = Agent("Agent1")
    agent2 = Agent("Agent2")
    agent1.start()
    agent2.start()

    # Simulate message exchange and threat detection
    while True:
        agent1.send_message("smart_city/alerts", "Threat detected!")
        agent2.send_message("smart_city/alerts", "Mitigation in progress")
