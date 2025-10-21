# SURG Architecture Documentation

This document provides detailed architectural diagrams and explanations for the SURG (Smart User Recommendation with GenAI) system.

## 🏗️ High-Level System Architecture

```mermaid
graph TB
    subgraph "Client Applications"
        WEB[Web App]
        API[REST API]
        MOBILE[Mobile App]
        BATCH[Batch Jobs]
    end
    
    subgraph "SURG Core System"
        GATEWAY[API Gateway]
        SURG_MAIN[SURG Engine]
        
        subgraph "Core Modules"
            CONFIG[Configuration]
            DATA[Data Module]
            RECOMMENDER[Recommender Engine]
            GENAI[GenAI Module]
            EVAL[Evaluation Module]
            EMBED[Embeddings Module]
        end
    end
    
    subgraph "External Services"
        OPENAI[OpenAI API]
        CUSTOM_LLM[Custom LLM]
        VECTOR_DB[Vector Database]
    end
    
    subgraph "Data Storage"
        PRIMARY_DB[(Primary Database)]
        CACHE[(Redis Cache)]
        FEATURE_STORE[(Feature Store)]
        MODEL_STORE[(Model Registry)]
    end
    
    subgraph "Infrastructure"
        MONITORING[Monitoring]
        LOGGING[Logging]
        METRICS[Metrics]
    end
    
    %% Client connections
    WEB --> GATEWAY
    API --> GATEWAY
    MOBILE --> GATEWAY
    BATCH --> GATEWAY
    
    %% Core system connections
    GATEWAY --> SURG_MAIN
    SURG_MAIN --> CONFIG
    SURG_MAIN --> DATA
    SURG_MAIN --> RECOMMENDER
    SURG_MAIN --> GENAI
    SURG_MAIN --> EVAL
    SURG_MAIN --> EMBED
    
    %% Data flow
    DATA --> PRIMARY_DB
    DATA --> CACHE
    DATA --> FEATURE_STORE
    
    %% AI/ML connections
    GENAI --> OPENAI
    GENAI --> CUSTOM_LLM
    EMBED --> VECTOR_DB
    RECOMMENDER --> MODEL_STORE
    
    %% Infrastructure
    SURG_MAIN --> MONITORING
    SURG_MAIN --> LOGGING
    SURG_MAIN --> METRICS
    
    %% Styling
    classDef client fill:#e1f5fe
    classDef core fill:#f3e5f5
    classDef external fill:#fff3e0
    classDef storage fill:#e8f5e8
    classDef infra fill:#fce4ec
    
    class WEB,API,MOBILE,BATCH client
    class SURG_MAIN,CONFIG,DATA,RECOMMENDER,GENAI,EVAL,EMBED core
    class OPENAI,CUSTOM_LLM,VECTOR_DB external
    class PRIMARY_DB,CACHE,FEATURE_STORE,MODEL_STORE storage
    class MONITORING,LOGGING,METRICS infra
```

## 🔄 Recommendation Generation Flow

```mermaid
sequenceDiagram
    participant User as End User
    participant App as Application
    participant SURG as SURG Engine
    participant Data as Data Module
    participant Rec as Recommender
    participant GenAI as GenAI Module
    participant Cache as Cache Layer
    participant DB as Database
    
    User->>App: Request recommendations
    App->>SURG: recommend(user_id, params)
    
    SURG->>Cache: Check cached recommendations
    alt Cache Hit
        Cache-->>SURG: Return cached results
    else Cache Miss
        SURG->>Data: Load user profile
        Data->>DB: Query user data
        DB-->>Data: User profile data
        Data-->>SURG: User context
        
        SURG->>Rec: Generate base recommendations
        Rec->>DB: Query interaction history
        DB-->>Rec: Historical data
        Rec->>Rec: Apply ML algorithms
        Rec-->>SURG: Base recommendations
        
        opt GenAI Enhancement Enabled
            SURG->>GenAI: Enhance recommendations
            GenAI->>GenAI: Apply LLM reasoning
            GenAI-->>SURG: Enhanced recommendations
        end
        
        SURG->>Cache: Store results
    end
    
    SURG-->>App: Return recommendations
    App-->>User: Display personalized results
    
    Note over User,DB: Total latency: 50-200ms (cached) or 500-2000ms (fresh)
```

## 🧠 GenAI Enhancement Pipeline

```mermaid
flowchart TD
    A[Base Recommendations] --> B{GenAI Enabled?}
    B -->|No| I[Return Base Results]
    B -->|Yes| C[Extract Context]
    
    C --> D[User Profile Analysis]
    C --> E[Item Feature Analysis]
    C --> F[Interaction Patterns]
    
    D --> G[Build Prompt]
    E --> G
    F --> G
    
    G --> H[LLM Processing]
    H --> J{Enhancement Type}
    
    J -->|Rerank| K[Reorder Recommendations]
    J -->|Filter| L[Remove Inappropriate Items]
    J -->|Explain| M[Generate Explanations]
    J -->|Augment| N[Add New Recommendations]
    
    K --> O[Combine Results]
    L --> O
    M --> O
    N --> O
    
    O --> P[Quality Check]
    P --> Q{Pass Quality Gate?}
    Q -->|Yes| R[Return Enhanced Results]
    Q -->|No| S[Fallback to Base Results]
    
    style H fill:#ffeb3b
    style O fill:#4caf50
    style P fill:#ff9800
```

## 📊 Data Flow Architecture

```mermaid
graph LR
    subgraph "Data Ingestion"
        STREAM[Real-time Stream]
        BATCH[Batch Import]
        API_DATA[API Data]
    end
    
    subgraph "Data Processing"
        VALIDATE[Data Validation]
        TRANSFORM[Data Transformation]
        ENRICH[Data Enrichment]
    end
    
    subgraph "Feature Engineering"
        USER_FEAT[User Features]
        ITEM_FEAT[Item Features]
        INTERACTION_FEAT[Interaction Features]
        CONTEXT_FEAT[Context Features]
    end
    
    subgraph "Storage Layer"
        RAW_DATA[(Raw Data Store)]
        PROCESSED_DATA[(Processed Data)]
        FEATURE_STORE[(Feature Store)]
        VECTOR_STORE[(Vector Store)]
    end
    
    subgraph "ML Pipeline"
        FEATURE_SELECT[Feature Selection]
        MODEL_TRAIN[Model Training]
        MODEL_EVAL[Model Evaluation]
        MODEL_DEPLOY[Model Deployment]
    end
    
    %% Data flow
    STREAM --> VALIDATE
    BATCH --> VALIDATE
    API_DATA --> VALIDATE
    
    VALIDATE --> TRANSFORM
    TRANSFORM --> ENRICH
    
    ENRICH --> USER_FEAT
    ENRICH --> ITEM_FEAT
    ENRICH --> INTERACTION_FEAT
    ENRICH --> CONTEXT_FEAT
    
    VALIDATE --> RAW_DATA
    ENRICH --> PROCESSED_DATA
    USER_FEAT --> FEATURE_STORE
    ITEM_FEAT --> FEATURE_STORE
    INTERACTION_FEAT --> FEATURE_STORE
    CONTEXT_FEAT --> FEATURE_STORE
    
    USER_FEAT --> VECTOR_STORE
    ITEM_FEAT --> VECTOR_STORE
    
    FEATURE_STORE --> FEATURE_SELECT
    FEATURE_SELECT --> MODEL_TRAIN
    MODEL_TRAIN --> MODEL_EVAL
    MODEL_EVAL --> MODEL_DEPLOY
    
    style VALIDATE fill:#ffcdd2
    style FEATURE_STORE fill:#c8e6c9
    style MODEL_DEPLOY fill:#bbdefb
```

## 🎯 Algorithm Selection Flow

```mermaid
flowchart TD
    A[Recommendation Request] --> B{Data Availability Check}
    
    B -->|No User History| C[Content-Based Filtering]
    B -->|Limited History| D[Hybrid Approach]
    B -->|Rich History| E[Algorithm Selection]
    
    E --> F{User Preference}
    F -->|Accuracy| G[Collaborative Filtering]
    F -->|Diversity| H[Content-Based]
    F -->|Novelty| I[Matrix Factorization]
    F -->|Explainability| J[GenAI Enhanced]
    
    C --> K[Execute Algorithm]
    D --> K
    G --> K
    H --> K
    I --> K
    J --> K
    
    K --> L{Performance Check}
    L -->|Good| M[Return Results]
    L -->|Poor| N[Fallback Strategy]
    
    N --> O[Simple Popularity-Based]
    O --> M
    
    M --> P{Post-Processing Needed?}
    P -->|Yes| Q[Apply Filters]
    P -->|No| R[Final Results]
    Q --> R
    
    style E fill:#e1bee7
    style K fill:#fff9c4
    style L fill:#ffcdd2
    style R fill:#c8e6c9
```

## 🔧 Configuration Management

```mermaid
graph TB
    subgraph "Configuration Sources"
        ENV[Environment Variables]
        FILE[Config Files]
        DB_CONFIG[Database Config]
        RUNTIME[Runtime Parameters]
    end
    
    subgraph "Configuration Layers"
        DEFAULT[Default Values]
        USER[User Overrides]
        ENVIRONMENT[Environment Specific]
        DYNAMIC[Dynamic Updates]
    end
    
    subgraph "Configuration Categories"
        ALG_CONFIG[Algorithm Settings]
        GENAI_CONFIG[GenAI Settings]
        DATA_CONFIG[Data Settings]
        PERF_CONFIG[Performance Settings]
        SEC_CONFIG[Security Settings]
    end
    
    subgraph "Validation & Processing"
        VALIDATE_CONFIG[Config Validation]
        MERGE[Config Merging]
        APPLY[Apply Settings]
    end
    
    %% Sources to layers
    ENV --> DEFAULT
    FILE --> USER
    DB_CONFIG --> ENVIRONMENT
    RUNTIME --> DYNAMIC
    
    %% Layer hierarchy
    DEFAULT --> MERGE
    USER --> MERGE
    ENVIRONMENT --> MERGE
    DYNAMIC --> MERGE
    
    %% Processing
    MERGE --> VALIDATE_CONFIG
    VALIDATE_CONFIG --> ALG_CONFIG
    VALIDATE_CONFIG --> GENAI_CONFIG
    VALIDATE_CONFIG --> DATA_CONFIG
    VALIDATE_CONFIG --> PERF_CONFIG
    VALIDATE_CONFIG --> SEC_CONFIG
    
    %% Application
    ALG_CONFIG --> APPLY
    GENAI_CONFIG --> APPLY
    DATA_CONFIG --> APPLY
    PERF_CONFIG --> APPLY
    SEC_CONFIG --> APPLY
    
    style VALIDATE_CONFIG fill:#ffcdd2
    style MERGE fill:#fff9c4
    style APPLY fill:#c8e6c9
```

## 🚀 Deployment Architecture

```mermaid
graph TB
    subgraph "Load Balancer Layer"
        LB[Load Balancer]
        CDN[CDN/Edge Cache]
    end
    
    subgraph "Application Layer"
        API1[SURG API Instance 1]
        API2[SURG API Instance 2]
        API3[SURG API Instance N]
        
        WORKER1[Background Worker 1]
        WORKER2[Background Worker 2]
    end
    
    subgraph "Caching Layer"
        REDIS_MASTER[(Redis Master)]
        REDIS_SLAVE[(Redis Slave)]
        MEMCACHED[(Memcached)]
    end
    
    subgraph "Database Layer"
        PG_PRIMARY[(PostgreSQL Primary)]
        PG_REPLICA[(PostgreSQL Replica)]
        VECTOR_DB[(Vector Database)]
    end
    
    subgraph "External Services"
        OPENAI_API[OpenAI API]
        MONITORING_SVC[Monitoring Service]
        LOGGING_SVC[Logging Service]
    end
    
    subgraph "Message Queue"
        RABBITMQ[RabbitMQ]
        CELERY[Celery Workers]
    end
    
    %% Load balancing
    LB --> API1
    LB --> API2
    LB --> API3
    CDN --> LB
    
    %% API to services
    API1 --> REDIS_MASTER
    API2 --> REDIS_MASTER
    API3 --> REDIS_MASTER
    
    API1 --> PG_PRIMARY
    API2 --> PG_REPLICA
    API3 --> PG_REPLICA
    
    API1 --> VECTOR_DB
    API2 --> VECTOR_DB
    API3 --> VECTOR_DB
    
    %% Background processing
    API1 --> RABBITMQ
    API2 --> RABBITMQ
    API3 --> RABBITMQ
    
    RABBITMQ --> WORKER1
    RABBITMQ --> WORKER2
    WORKER1 --> CELERY
    WORKER2 --> CELERY
    
    %% Replication
    REDIS_MASTER --> REDIS_SLAVE
    PG_PRIMARY --> PG_REPLICA
    
    %% External services
    API1 --> OPENAI_API
    API2 --> OPENAI_API
    API3 --> OPENAI_API
    
    API1 --> MONITORING_SVC
    API2 --> MONITORING_SVC
    API3 --> MONITORING_SVC
    
    API1 --> LOGGING_SVC
    API2 --> LOGGING_SVC
    API3 --> LOGGING_SVC
    
    style LB fill:#e3f2fd
    style API1,API2,API3 fill:#f3e5f5
    style REDIS_MASTER,PG_PRIMARY fill:#e8f5e8
    style OPENAI_API fill:#fff3e0
```

## 📈 Performance Monitoring Flow

```mermaid
flowchart LR
    A[Request] --> B[Metrics Collection]
    B --> C[Latency Measurement]
    B --> D[Throughput Tracking]
    B --> E[Error Rate Monitoring]
    B --> F[Resource Usage]
    
    C --> G[Time Series DB]
    D --> G
    E --> G
    F --> G
    
    G --> H[Dashboard]
    G --> I[Alerting System]
    G --> J[Analytics Engine]
    
    I --> K{Threshold Exceeded?}
    K -->|Yes| L[Send Alert]
    K -->|No| M[Continue Monitoring]
    
    J --> N[Performance Analysis]
    N --> O[Optimization Recommendations]
    
    H --> P[Real-time Visualization]
    L --> Q[Incident Response]
    
    style B fill:#fff3e0
    style G fill:#e8f5e8
    style I fill:#ffcdd2
    style P fill:#e1f5fe
```

## 🔒 Security Architecture

```mermaid
graph TB
    subgraph "External Threats"
        DDOS[DDoS Attacks]
        INJECTION[SQL Injection]
        XSS[Cross-Site Scripting]
        API_ABUSE[API Abuse]
    end
    
    subgraph "Security Layers"
        WAF[Web Application Firewall]
        RATE_LIMIT[Rate Limiting]
        AUTH[Authentication]
        AUTHZ[Authorization]
        ENCRYPTION[Data Encryption]
    end
    
    subgraph "Security Controls"
        INPUT_VAL[Input Validation]
        OUTPUT_ENCODE[Output Encoding]
        CSRF_PROTECT[CSRF Protection]
        API_KEY[API Key Management]
        AUDIT_LOG[Audit Logging]
    end
    
    subgraph "Data Protection"
        PII_MASK[PII Masking]
        DATA_ENCRYPT[Data Encryption at Rest]
        TRANSIT_ENCRYPT[Encryption in Transit]
        BACKUP_ENCRYPT[Encrypted Backups]
    end
    
    %% Threat mitigation
    DDOS --> WAF
    INJECTION --> INPUT_VAL
    XSS --> OUTPUT_ENCODE
    API_ABUSE --> RATE_LIMIT
    
    %% Security layers
    WAF --> AUTH
    RATE_LIMIT --> AUTH
    AUTH --> AUTHZ
    AUTHZ --> ENCRYPTION
    
    %% Controls
    INPUT_VAL --> CSRF_PROTECT
    OUTPUT_ENCODE --> API_KEY
    CSRF_PROTECT --> AUDIT_LOG
    
    %% Data protection
    ENCRYPTION --> PII_MASK
    PII_MASK --> DATA_ENCRYPT
    DATA_ENCRYPT --> TRANSIT_ENCRYPT
    TRANSIT_ENCRYPT --> BACKUP_ENCRYPT
    
    style WAF fill:#ffcdd2
    style AUTH fill:#fff3e0
    style ENCRYPTION fill:#e8f5e8
    style AUDIT_LOG fill:#e1f5fe
```

This comprehensive architecture documentation provides multiple perspectives on the SURG system, from high-level overviews to detailed implementation flows. Each diagram serves a specific purpose in understanding different aspects of the system's design and operation.