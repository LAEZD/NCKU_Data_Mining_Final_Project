# Unified Input Builder Demo
# =========================
# 演示新的統一輸入構建策略如何工作
# 展示元數據如何作為特權文本被注入到輸入序列中

import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import sys
import os

# 添加 preprocessing 模組的路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.unified_input_builder import UnifiedInputBuilder
from preprocessing.metadata_features import MetadataFeatures

def demo_unified_input_builder():
    """
    演示統一輸入構建器的功能
    """
    print("=" * 60)
    print("統一輸入構建器演示")
    print("=" * 60)
    
    # 初始化 tokenizer
    print("\n1. 初始化 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 創建示例數據
    print("\n2. 創建示例數據...")
    sample_data = {
        'prompt': "Which response is better for a customer service scenario?",
        'response_a': "Thank you for contacting us. I understand your concern and will help you resolve this issue immediately. Let me check your account details and provide a comprehensive solution.",
        'response_b': "Thanks for calling. I'll help."
    }
    
    # 計算元數據特徵
    print("\n3. 計算元數據特徵...")
    row = pd.Series(sample_data)
    metadata_features = MetadataFeatures.extract_core_features(row)
    print(f"   元數據特徵: {metadata_features}")
    
    # 測試不同的輸入構建策略
    print("\n4. 測試不同的輸入構建策略...")
    
    # 4.1 不包含元數據的傳統方式
    print("\n   4.1 傳統方式（無元數據）:")
    traditional_input = UnifiedInputBuilder.create_unified_input(
        tokenizer=tokenizer,
        prompt=sample_data['prompt'],
        response_a=sample_data['response_a'],
        response_b=sample_data['response_b'],
        metadata_dict=None,
        max_len=128
    )
    
    traditional_text = tokenizer.decode(traditional_input['input_ids'], skip_special_tokens=False)
    print(f"      生成的文本: {traditional_text[:200]}...")
    print(f"      序列長度: {len(traditional_input['input_ids'])}")
    print(f"      有效 token 數: {sum(traditional_input['attention_mask'])}")
    
    # 4.2 包含元數據的新方式
    print("\n   4.2 新統一方式（包含元數據）:")
    unified_input = UnifiedInputBuilder.create_unified_input(
        tokenizer=tokenizer,
        prompt=sample_data['prompt'],
        response_a=sample_data['response_a'],
        response_b=sample_data['response_b'],
        metadata_dict=metadata_features,
        max_len=128
    )
    
    unified_text = tokenizer.decode(unified_input['input_ids'], skip_special_tokens=False)
    print(f"      生成的文本: {unified_text[:200]}...")
    print(f"      序列長度: {len(unified_input['input_ids'])}")
    print(f"      有效 token 數: {sum(unified_input['attention_mask'])}")
    
    # 4.3 元數據字串格式化測試
    print("\n   4.3 元數據字串格式化:")
    metadata_string = UnifiedInputBuilder._format_metadata_string(metadata_features)
    print(f"      原始元數據: {metadata_features}")
    print(f"      格式化字串: '{metadata_string}'")
    print(f"      字串長度: {len(metadata_string)} 字符")
    print(f"      Token 化後長度: {len(tokenizer.encode(metadata_string, add_special_tokens=False))} tokens")
    
    # 5. 測試極端情況
    print("\n5. 測試極端情況...")
    
    # 5.1 非常長的輸入
    print("\n   5.1 超長輸入測試:")
    long_prompt = "This is a very long prompt. " * 50
    long_response_a = "This is response A repeated many times. " * 30
    long_response_b = "Response B is also very long and repeated. " * 25
    
    long_input = UnifiedInputBuilder.create_unified_input(
        tokenizer=tokenizer,
        prompt=long_prompt,
        response_a=long_response_a,
        response_b=long_response_b,
        metadata_dict=metadata_features,
        max_len=128
    )
    
    print(f"      超長輸入處理後的有效 token 數: {sum(long_input['attention_mask'])}")
     
    # 5.2 元數據優先級測試
    print("\n   5.2 元數據優先級測試:")
    # 創建大量元數據特徵
    extensive_metadata = {
        'punc_v_a': 5, 'punc_v_b': 3, 'resp_jaccard': 0.75, 'len_ratio': 2.4,
        'prompt_length': 100, 'response_a_length': 200, 'response_b_length': 50,
        'total_length': 350, 'punc_v_prompt': 8
    }
    
    priority_input = UnifiedInputBuilder.create_unified_input(
        tokenizer=tokenizer,
        prompt=long_prompt,
        response_a=long_response_a,
        response_b=long_response_b,
        metadata_dict=extensive_metadata,
        max_len=128
    )
    
    priority_text = tokenizer.decode(priority_input['input_ids'], skip_special_tokens=False)
    print(f"      優先級處理後的文本開頭: {priority_text[:100]}...")
    
    # 檢查元數據是否被保留
    extensive_metadata_string = UnifiedInputBuilder._format_metadata_string(extensive_metadata)
    metadata_preserved = extensive_metadata_string in priority_text
    print(f"      元數據是否完整保留: {metadata_preserved}")
    if metadata_preserved:
        print("      ✅ 元數據成功作為特權文本被完整保留！")
    else:
        print("      ⚠️  元數據可能被截斷或處理不當")
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)
    
    return {
        'traditional_input': traditional_input,
        'unified_input': unified_input,
        'priority_input': priority_input,
        'metadata_features': metadata_features
    }

def compare_input_strategies():
    """
    比較不同輸入策略的效果
    """
    print("\n" + "=" * 60)
    print("輸入策略比較分析")
    print("=" * 60)
    
    # 運行演示
    results = demo_unified_input_builder()
    
    print("\n策略比較總結:")
    print("-" * 40)
    print("1. 傳統方式（無元數據）:")
    print("   ✗ 無法利用元數據信息")
    print("   ✗ 模型無法學習數據的結構化特徵")
    print("   ✓ 更多空間給文本內容")
    
    print("\n2. 舊方式（元數據作為並行特徵）:")
    print("   ✗ 標準 Transformer 模型無法使用")
    print("   ✗ 需要自定義模型架構")
    print("   ✗ 增加複雜性")
    
    print("\n3. 新統一方式（元數據作為特權文本）:")
    print("   ✓ 完全兼容標準 Transformer 模型")
    print("   ✓ 元數據作為特權文本優先保留")
    print("   ✓ 智能的預算分配策略")
    print("   ✓ 模型能學習元數據和文本的聯合表示")
    print("   ✓ 無需修改下游模型架構")
    
    print("\n優勢總結:")
    print("-" * 40)
    print("• 數據-模型匹配: 解決了元數據無法被模型使用的根本問題")
    print("• Token 預算管理: 智能分配有限的 token 空間")
    print("• 簡化架構: 保持與標準 Trainer 的完全兼容")
    print("• 優先級策略: 確保最重要的元數據信息不會丟失")
    print("• 靈活性: 支持不同類型和數量的元數據特徵")

def test_improved_unified_input():
    """
    測試改進後的統一輸入構建策略
    """
    print("="*80)
    print("測試改進後的統一輸入構建策略")
    print("="*80)
    
    # 1. 初始化 tokenizer
    print("\n1. 初始化 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 創建各種測試場景
    test_scenarios = [
        {
            "name": "常規場景",
            "prompt": "Which response is better for customer service?",
            "response_a": "Thank you for contacting us. I understand your concern.",
            "response_b": "OK.",
            "metadata": {'pa': 1, 'pb': 0, 'jc': 0.3, 'lr': 15.2},
            "max_len": 128
        },
        {
            "name": "元數據豐富場景", 
            "prompt": "Evaluate these responses for quality and helpfulness.",
            "response_a": "This is a comprehensive response with detailed explanations.",
            "response_b": "Short answer.",
            "metadata": {
                'pa': 5, 'pb': 1, 'jc': 0.75, 'lr': 8.5,
                'pl': 45, 'al': 120, 'bl': 25, 'wo': 0.6, 'co': 0.8
            },
            "max_len": 128
        },
        {
            "name": "極長輸入場景",
            "prompt": "This is a very long prompt that contains multiple sentences and detailed context about the situation that needs to be evaluated carefully by considering all aspects and nuances of the problem statement.",
            "response_a": "This is an extremely detailed response that covers all possible angles of the question, provides comprehensive explanations, includes multiple examples, and ensures complete understanding of the topic through thorough analysis and clear communication.",
            "response_b": "This is another very detailed response that also aims to be comprehensive and helpful by providing extensive information, detailed examples, and thorough explanations to ensure the user gets complete understanding.",
            "metadata": {'pa': 3, 'pb': 2, 'jc': 0.85, 'lr': 1.1},
            "max_len": 128
        },
        {
            "name": "超緊湊場景",
            "prompt": "Rate responses.",
            "response_a": "Good.",
            "response_b": "Bad.",
            "metadata": {'pa': 0, 'pb': 0, 'jc': 0.0, 'lr': 1.0},
            "max_len": 64
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. 測試場景：{scenario['name']}")
        print("-" * 50)
        
        # 測試改進後的統一輸入構建
        result = UnifiedInputBuilder.create_unified_input(
            tokenizer=tokenizer,
            prompt=scenario['prompt'],
            response_a=scenario['response_a'],
            response_b=scenario['response_b'],
            metadata_dict=scenario['metadata'],
            max_len=scenario['max_len']
        )
        
        # 分析結果
        input_ids = result['input_ids'].numpy()
        attention_mask = result['attention_mask'].numpy()
        
        # 計算有效 token 數量
        effective_tokens = sum(attention_mask)
        
        # 解碼文本來檢查結果
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        
        print(f"   序列長度: {len(input_ids)}")
        print(f"   有效 tokens: {effective_tokens}")
        print(f"   生成的文本前150字符: {decoded_text[:150]}...")
        
        # 檢查元數據是否被正確包含
        metadata_str = UnifiedInputBuilder._format_metadata_string(scenario['metadata'])
        print(f"   元數據字串: '{metadata_str}'")
        print(f"   元數據字串長度: {len(metadata_str)} 字符")
        
        # 檢查截斷情況
        if effective_tokens == scenario['max_len']:
            print("   ⚠️  序列已達到最大長度限制")
        else:
            print(f"   ✓ 序列長度正常 ({effective_tokens}/{scenario['max_len']})")
    
    print(f"\n{'='*80}")
    print("測試完成！改進後的統一輸入構建器可以更好地：")
    print("1. 使用極短的元數據鍵名 (pa, pb, jc, lr)")
    print("2. 智能格式化數值以節省空間")
    print("3. 動態分配 token 預算")
    print("4. 優先保護重要信息")
    print("5. 提供詳細的分配統計")
    print("="*80)

# 更新主函數以包含新的測試
if __name__ == "__main__":
    # 運行原始演示
    demo_unified_input_builder()
    
    print("\n" + "="*100 + "\n")
    
    # 運行改進後的測試
    test_improved_unified_input()
