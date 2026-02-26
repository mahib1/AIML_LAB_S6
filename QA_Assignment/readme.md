## Cloud Computing Questions and Answers

**Q1: What is the primary difference between Scalability and Elasticity in a cloud environment?**
**Answer:** Scalability is the ability of a system to handle a growing amount of work by adding resources (up or out) permanently or semi-permanently. Elasticity is the ability to automatically scale those resources up or down in real-time based on immediate demand spikes or drops.

**Q2: Describe the three main Cloud Service Models: IaaS, PaaS, and SaaS.**
**Answer:** IaaS (Infrastructure as a Service) provides virtualized computing resources like servers and storage. PaaS (Platform as a Service) provides a framework for developers to build and deploy apps without managing hardware. SaaS (Software as a Service) delivers ready-to-use software over the internet.

**Q3: What role does a Hypervisor play in cloud infrastructure?**
**Answer:** A Hypervisor is software that creates and runs Virtual Machines (VMs). It sits between the physical hardware and the VMs, allowing multiple operating systems to share the same physical resources safely.

**Q4: What is "Cloud Bursting" and what problem does it solve?**
**Answer:** Cloud Bursting is a configuration where an application runs in a private cloud but "bursts" into a public cloud when demand peaks. This prevents service outages during high traffic while keeping baseline costs low in the private cloud.

**Q5: Contrast Public, Private, and Hybrid cloud deployment models.**
**Answer:** Public clouds are owned by providers (like AWS) and shared by many users. Private clouds are dedicated to a single organization. Hybrid clouds combine both, allowing data and apps to move between them for better flexibility and security.

**Q6: In Serverless computing (FaaS), what is a "Cold Start"?**
**Answer:** A Cold Start occurs when a serverless function is triggered after being idle. Because the cloud provider has spun down the resources, it takes extra time to initialize the environment and run the code, causing latency for that first request.

**Q7: How does Edge Computing complement Cloud Computing?**
**Answer:** Edge Computing processes data closer to the source (like IoT devices) to reduce latency and bandwidth usage. It handles immediate, time-sensitive tasks, while the central Cloud handles long-term storage and heavy analytical processing.

**Q8: What are the core pillars of "Cloud-Native" application architecture?**
**Answer:** The pillars include Microservices (modular components), Containers (like Docker for portability), Orchestration (like Kubernetes), and DevOps practices like Continuous Integration/Continuous Deployment (CI/CD).

**Q9: How can an organization mitigate "Vendor Lock-in" when using cloud services?**
**Answer:** Organizations can use multi-cloud strategies, adopt open-source standards, use "Infrastructure as Code" (IaC) tools like Terraform, and employ containerization to ensure applications can be moved between providers with minimal friction.

**Q10: Explain the "Shared Responsibility Model" regarding cloud security.**
**Answer:** The provider is responsible for the security *of* the cloud (physical centers, hardware, and global infrastructure). The customer is responsible for security *in* the cloud (data encryption, identity management, and patching the guest operating systems).

---

## libp2p (Peer-to-Peer Networking) Questions and Answers

**Q1: What exactly is libp2p?**
**Answer:** libp2p is a modular, peer-to-peer networking stack and library. It provides a standardized way for decentralized applications to handle discovery, routing, and communication across different network protocols.

**Q2: What is a Multiaddr and why is it used?**
**Answer:** A Multiaddr is a self-describing network address. It includes the transport protocol and the address (e.g., `/ip4/127.0.0.1/tcp/4001`). This allows nodes to understand exactly how to reach a peer regardless of the underlying technology.

**Q3: How is a Peer ID generated in a libp2p network?**
**Answer:** A Peer ID is a unique identifier for a node, created by hashing the node’s public key. This ensures the node's identity remains consistent even if its IP address or location changes.

**Q4: What is the function of the "Identify" protocol in libp2p?**
**Answer:** When two peers connect, they use the Identify protocol to exchange essential information, such as their Peer IDs, public keys, and the specific protocols they are capable of handling (e.g., DHT or PubSub).

**Q5: Explain how "Transports" work in libp2p.**
**Answer:** Transports are the modules responsible for sending and receiving raw data. libp2p is transport-agnostic, meaning it can run over TCP, UDP, QUIC, WebSockets, or even specialized transports like Bluetooth or Tor.

**Q6: What is "Circuit Relay" and why is it necessary?**
**Answer:** Circuit Relay is a technique where a third-party node acts as a proxy to connect two peers that cannot communicate directly due to strict firewalls or NAT (Network Address Translation).

**Q7: How does the Kademlia DHT help with peer routing?**
**Answer:** The Distributed Hash Table (DHT) allows nodes to find other nodes or specific data by using an XOR-based distance metric. It creates a structured way for a node to ask, "Who is closest to this Peer ID/Data Hash?" until the target is found.

**Q8: Describe the "Gossipsub" protocol.**
**Answer:** Gossipsub is a publish-subscribe protocol that uses a "mesh" for efficient message delivery and "gossip" (metadata exchange) to ensure the network stays resilient and messages reach all subscribers without flooding the entire network.

**Q9: What happens during a "Connection Upgrade" in libp2p?**
**Answer:** A connection upgrade takes a raw byte stream (like a TCP connection) and layers additional functionality onto it, such as security (TLS or Noise) and stream multiplexing (allowing many independent streams over one connection).

**Q10: What is Content-Addressable networking in the context of libp2p-based systems?**
**Answer:** It is a system where data is retrieved based on its content (the hash/CID) rather than its location (an IP). libp2p facilitates this by allowing the network to route requests to whichever peer currently holds the specific hash.

---

## LLM Security Questions and Answers

**Q1: What is "Prompt Injection"?**
**Answer:** Prompt Injection is an attack where a user provides input designed to hijack the LLM’s control flow. It forces the model to ignore its original system instructions and perform unintended or malicious actions.

**Q2: Define "Data Leakage" in Large Language Models.**
**Answer:** Data leakage occurs when an LLM inadvertently reveals sensitive or private information (like passwords or PII) that was included in its training data or fine-tuning set during a conversation with a user.

**Q3: What constitutes an "Adversarial Attack" on an AI model?**
**Answer:** An adversarial attack involves crafting specific, often hidden or subtle, inputs that cause the model to malfunction, produce incorrect results, or bypass safety filters it would normally follow.

**Q4: How does "Indirect Prompt Injection" differ from standard injection?**
**Answer:** In an indirect injection, the malicious instructions come from an external source the LLM is "reading" (like a website or email) rather than the user’s direct prompt. For example, a hidden instruction on a webpage could tell a browsing AI to steal the user's cookies.

**Q5: What is "Model Inversion" or "Extraction"?**
**Answer:** This is a privacy attack where an adversary queries the model repeatedly to reconstruct parts of the training data or the model's internal parameters, potentially exposing proprietary or sensitive information.

**Q6: What is the purpose of "AI Guardrails"?**
**Answer:** Guardrails are an independent layer of software that monitors both the input to and the output from an LLM. They filter out malicious prompts and prevent the model from outputting harmful, biased, or restricted content.

**Q7: Explain "Training Data Poisoning."**
**Answer:** This is an attack where malicious data is intentionally inserted into the training or fine-tuning dataset. This can create "backdoors" where the model behaves normally until it sees a specific trigger word, at which point it executes a malicious command.

**Q8: Why is "PII Scrubbing" critical in the LLM pipeline?**
**Answer:** PII (Personally Identifiable Information) scrubbing removes names, addresses, and other identifiers from data before training or fine-tuning. This prevents the model from "learning" and potentially leaking people's private details later.

**Q9: What are the security risks associated with "Agentic LLMs" (LLMs with tool access)?**
**Answer:** If an LLM has access to tools like a terminal, database, or email API, a prompt injection can lead to Remote Code Execution (RCE) or unauthorized data deletion. The risk is high because the AI can take physical actions in the digital world.

**Q10: Describe a "Membership Inference Attack" (MIA).**
**Answer:** This attack determines if a specific piece of data was part of the model's training set. Attackers do this by checking if the model responds with unusually high confidence or low "loss" for that specific data point, which signals familiarity.
