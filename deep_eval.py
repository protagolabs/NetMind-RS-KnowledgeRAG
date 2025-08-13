from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric  # 导入评估器  
from deepeval.test_case import LLMTestCase  # 导入测试用例
from dotenv import load_dotenv  # 导入dotenv
import json  # 导入json

load_dotenv()

class DeepEval:
    def __init__(self, question=None, retrieval_context=None, actual_output=None, expected_output=None):
        """
        Args:
            question: str, the question to be evaluated
            retrieval_context: list, the retrieval context
            actual_output: str, the actual output
            expected_output: str, the expected output
        """
        self.question = question
        self.retrieval_context = retrieval_context if retrieval_context is not None else []
        self.actual_output = actual_output
        self.expected_output = expected_output
        if not self.question and not self.actual_output:
            raise ValueError("question and actual_output can not be None.")
        self.result = {}  # 存储评估结果

    def evaluate(self):
        """
        Evaluate the question and actual output.
        """
        if self.question and self.actual_output:  # 如果问题和实际输出不为空，则评估答案相关性
            self.answer_relevancy()
        if self.retrieval_context and self.actual_output and self.question:  # 如果检索上下文、实际输出和问题不为空，则评估忠实度
            self.faithfulness()
        if self.question and self.retrieval_context and self.actual_output:  # 如果问题、检索上下文和实际输出不为空，则评估上下文相关性
            self.contextual_relevance()
        if self.question and self.retrieval_context and self.actual_output and self.expected_output:  # 如果问题、检索上下文、实际输出和期望输出不为空，则评估上下文精确度和召回率
            self.contextual_precision()
            self.contextual_recall()
        return self.result

    def answer_relevancy(self):
        """
        Evaluate the answer relevancy.
        """
        metric= AnswerRelevancyMetric(
            threshold=0.7,
            model="gpt-4.1",
            include_reason=True,
            # verbose_mode=True
        )
        test_case = LLMTestCase(  # 创建测试用例
            input=self.question,
            actual_output=self.actual_output
        )
        metric.measure(test_case)  # 评估答案相关性
        self.result["answer_relevancy"] = metric.score  # 存储答案相关性得分

    def faithfulness(self):
        """
        Evaluate the faithfulness.
        """
        metric= FaithfulnessMetric(  # 创建忠实度评估器
            threshold=0.7,
            model="gpt-4.1",
            include_reason=True,
            # verbose_mode=True
        )
        test_case = LLMTestCase(  # 创建测试用例
            input=self.question,
            retrieval_context=self.retrieval_context,
            actual_output=self.actual_output
        )
        metric.measure(test_case)  # 评估忠实度
        self.result["faithfulness"] = metric.score  # 存储忠实度得分

    def contextual_relevance(self):
        """
        Evaluate the contextual relevance.
        """
        metric= ContextualRelevancyMetric(  # 创建上下文相关性评估器
            threshold=0.7,
            model="gpt-4.1",
            include_reason=True,
            # verbose_mode=True
        )
        test_case = LLMTestCase(  # 创建测试用例
            input=self.question,
            retrieval_context=self.retrieval_context,
            actual_output=self.actual_output
        )
        metric.measure(test_case)  # 评估上下文相关性
        self.result["contextual_relevance"] = metric.score  # 存储上下文相关性得分

    def contextual_precision(self):
        """
        Evaluate the contextual precision.
        """
        metric= ContextualPrecisionMetric(  # 创建上下文精确度评估器
            threshold=0.7,
            model="gpt-4.1",
            include_reason=True,
            # verbose_mode=True
        )
        test_case = LLMTestCase(
            input=self.question,
            retrieval_context=self.retrieval_context,
            actual_output=self.actual_output,
            expected_output=self.expected_output
        )
        metric.measure(test_case)  # 评估上下文精确度
        self.result["contextual_precision"] = metric.score  # 存储上下文精确度得分

    def contextual_recall(self):
        """
        Evaluate the contextual recall.
        """
        metric= ContextualRecallMetric(  # 创建上下文召回率评估器
            threshold=0.7,
            model="gpt-4.1",
            include_reason=True,
            # verbose_mode=True
        )
        test_case = LLMTestCase(  # 创建测试用例
            input=self.question,
            retrieval_context=self.retrieval_context,
            actual_output=self.actual_output,
            expected_output=self.expected_output
        )
        metric.measure(test_case)  # 评估上下文召回率
        self.result["contextual_recall"] = metric.score  # 存储上下文召回率得分

if __name__ == "__main__":
    eval=DeepEval(  # 创建测试用例
        question="What is the capital of France?", 
        actual_output="Paris",
        retrieval_context=["Paris is the capital of France."],
        expected_output="Paris"
    )
    result=eval.evaluate()  # 评估测试用例
    print(json.dumps(result, indent=4))  # 打印评估结果


