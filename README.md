graph LR
    subgraph Edge_Layer [⌚ Edge Devices]
        A[Wearable Sensors] -- "BPM & Activity Data" --> B[Azure IoT Hub]
    end

    subgraph Azure_Cloud [☁️ Microsoft Azure Cloud]
        direction TB
        B --> C{ShiftGuard Engine}
        
        %% The Brain
        C -- "1. Narrative Triage" --> D[Azure AI Language]
        D -.->|"Sentiment & Entities"| C
        
        %% The Memory
        C -- "2. Risk & Audit Logs" --> E[(Azure SQL Database)]
        
        %% The Action
        C -- "3. Trigger Alert" --> F[Azure Logic Apps]
    end

    subgraph Action_Layer [📢 Notification]
        F --> G[Microsoft Teams / Discord]
        F --> H[Shift Manager Email]
    end

    %% Styling
    style Edge_Layer fill:#f9f9f9,stroke:#333,stroke-width:2px
    style Azure_Cloud fill:#e1f5fe,stroke:#0078d4,stroke-width:2px
    style Action_Layer fill:#f0f4c3,stroke:#827717,stroke-width:2px
    style C fill:#ffcc80,stroke:#e65100,stroke-width:2px,color:black
    style D fill:#b3e5fc,stroke:#01579b,stroke-width:2px
    style E fill:#b3e5fc,stroke:#01579b,stroke-width:2px
