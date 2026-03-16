# 大语言模型 (LLM) 推理工作流

这是一张展示一个词（Token）从输入大模型，到最终生成下一个词的完整流程图。

```mermaid
graph TD
    %% 阶段 1：输入与准备
    subgraph 阶段 1: 文本输入与查表 (准备阶段)
        A[输入文本: "我喜欢吃苹果"] --> B(切词 Tokenization)
        B --> C{词表映射字典}
        C -->|例如: 苹果 -> 401| D[Token ID: 401]
        D --> E[(词向量库 Embedding)]
        E -->|查表得到初始绝对坐标| F[初始词向量 X <br> e.g. 0.85, 0.80, 0.00]
        
        %% 位置编码：让模型知道词的顺序
        PE[位置编码 Positional Encoding <br> 告诉模型这是第5个词] -->|相加| F
    end

    %% 阶段 2：Transformer 核心计算
    subgraph 阶段 2: Transformer 层 (核心思考阶段, 重复 N 层)
        F --> G{第 1 层 Transformer}
        
        %% 自注意力机制
        subgraph 自注意力机制 (Self-Attention)
            G -->|复制 3 份| H1(乘以 W_Q 矩阵)
            G -->|复制 3 份| H2(乘以 W_K 矩阵)
            G -->|复制 3 份| H3(乘以 W_V 矩阵)
            
            H1 -->|生成| Q[Q: 查询向量 <br> 想找动词]
            H2 -->|生成| K[K: 键向量 / 存入 KV Cache <br> 我是名词/食物]
            H3 -->|生成| V[V: 值向量 / 存入 KV Cache <br> 苹果的实体概念]
            
            Q -->|与上下文中所有词的 K 计算点积/余弦相似度| AttentionScore(注意力分数/相关度排名)
            K -.-|被其他词的 Q 查询| AttentionScore
            
            AttentionScore -->|按照分数加权混合所有词的 V| ContextVector[融合了上下文的新向量]
            V -.-|提供实际内容| ContextVector
        end
        
        %% 前馈神经网络
        ContextVector --> FFN[前馈神经网络 FFN <br> 进一步提炼和映射特征]
        FFN --> Output1[第 1 层输出的新向量 X']
        
        Output1 -->|进入下一层...| LayerN{第 N 层 Transformer}
        LayerN -->|重复上述 QKV 融合| FinalVector[终极多维向量 <br> 包含了整句话极度深刻的语境]
    end

    %% 阶段 3：预测与输出
    subgraph 阶段 3: 预测下一个词 (输出阶段)
        FinalVector -->|投射回词表维度| Logits[全词表得分矩阵 <br> 包含 10万个词的打分]
        Logits --> Softmax(Softmax 函数 <br> 将得分转化为 0-100% 的概率)
        
        Softmax -->|概率排名前几的词| Candidates[候选词:<br>， 40%<br>。 30%<br>真 15%]
        
        Candidates -->|结合 Temperature 随机掷骰子| NextToken[选出下一个 Token: "，"]
        
        NextToken -->|将新词拼接到原句末尾| A
    end

    %% 样式调整
    classDef highlight fill:#f9f,stroke:#333,stroke-width:2px;
    class Q,K,V,ContextVector,FinalVector highlight;
```
