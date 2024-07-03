from ..settings.config import DEFAULT_MEMORY_KEY 
from NeoGPT.neogpt.prompts.prompt import stepback_prompt, PromptTemplate, ConversationBufferWindowMemory, INSTRUCTION_TEMPLATE


def test_stepback_prompt():
    # Test case 1
    model_type = "GPT3.5"
    persona = "default"
    memory_key = DEFAULT_MEMORY_KEY

    prompt, memory = stepback_prompt(model_type, persona, memory_key)

    # Assert the prompt and memory objects are returned correctly
    assert isinstance(prompt, PromptTemplate)
    assert isinstance(memory, ConversationBufferWindowMemory)
    assert prompt.input_variables == ["normal_context", "step_back_context", "question"]
    assert prompt.template.strip() == INSTRUCTION_TEMPLATE.strip();
    assert memory.k == memory_key
    assert memory.return_messages is True
    assert memory.input_key == "question"
    assert memory.memory_key == "history"

    # Test case 2
    model_type = "gpt3.5"
    persona = "custom"
    memory_key = 5

    prompt, memory = stepback_prompt(model_type, persona, memory_key)

    # Assert the prompt and memory objects are returned correctly
    assert isinstance(prompt, PromptTemplate)
    assert isinstance(memory, ConversationBufferWindowMemory)
    assert prompt.input_variables == ["normal_context", "step_back_context", "question"]
    assert prompt.template.strip() == INSTRUCTION_TEMPLATE.strip()
    assert memory.k == memory_key
    assert memory.return_messages is True
    assert memory.input_key == "question"
    assert memory.memory_key == "history"from NeoGPT.neogpt.settings.config import DEFAULT_MEMORY_KEY
from NeoGPT.neogpt.prompts.prompt import stepback_prompt, PromptTemplate, ConversationBufferWindowMemory, INSTRUCTION_TEMPLATE

def test_stepback_prompt():
    # Test case 1
    model_type = "mistral"
    persona = "default"
    memory_key = DEFAULT_MEMORY_KEY

    prompt, memory = stepback_prompt(model_type, persona, memory_key)

    # Assert the prompt and memory objects are returned correctly
    assert isinstance(prompt, PromptTemplate)
    assert isinstance(memory, ConversationBufferWindowMemory)
    assert prompt.input_variables == ["normal_context", "step_back_context", "question"]
    assert prompt.template.strip() == INSTRUCTION_TEMPLATE.strip()
    assert memory.k == memory_key
    assert memory.return_messages is True
    assert memory.input_key == "question"
    assert memory.memory_key == "history"

    # Test case 2
    model_type = "gpt3"
    persona = "custom"
    memory_key = 5

    prompt, memory = stepback_prompt(model_type, persona, memory_key)

    # Assert the prompt and memory objects are returned correctly
    assert isinstance(prompt, PromptTemplate)
    assert isinstance(memory, ConversationBufferWindowMemory)
    assert prompt.input_variables == ["normal_context", "step_back_context", "question"]
    assert prompt.template.strip() == INSTRUCTION_TEMPLATE.strip()
    assert memory.k == memory_key
    assert memory.return_messages is True
    assert memory.input_key == "question"
    assert memory.memory_key == "history"def test_stepback_prompt():
    # Test case 1
    model_type = "mistral"
    persona = "default"
    memory_key = DEFAULT_MEMORY_KEY

    prompt, memory = stepback_prompt(model_type, persona, memory_key)

    # Assert the prompt and memory objects are returned correctly
    assert isinstance(prompt, PromptTemplate)
    assert isinstance(memory, ConversationBufferWindowMemory)
    assert prompt.input_variables == ["normal_context", "step_back_context", "question"]
    assert prompt.template.strip() == INSTRUCTION_TEMPLATE.strip()
    assert memory.k == memory_key
    assert memory.return_messages is True
    assert memory.input_key == "question"
    assert memory.memory_key == "history"

    # Test case 2
    model_type = "gpt3"
    persona = "custom"
    memory_key = 5

    prompt, memory = stepback_prompt(model_type, persona, memory_key)

    # Assert the prompt and memory objects are returned correctly
    assert isinstance(prompt, PromptTemplate)
    assert isinstance(memory, ConversationBufferWindowMemory)
    assert prompt.input_variables == ["normal_context", "step_back_context", "question"]
    assert prompt.template.strip() == INSTRUCTION_TEMPLATE.strip()
    assert memory.k == memory_key
    assert memory.return_messages is True
    assert memory.input_key == "question"
    assert memory.memory_key == "history"def test_stepback_prompt():
    # Test case 1
    model_type = "mistral"
    persona = "default"
    memory_key = DEFAULT_MEMORY_KEY

    prompt, memory = stepback_prompt(model_type, persona, memory_key)

    # Assert the prompt and memory objects are returned correctly
    assert isinstance(prompt, PromptTemplate)
    assert isinstance(memory, ConversationBufferWindowMemory)
    assert prompt.input_variables == ["normal_context", "step_back_context", "question"]
    assert prompt.template.strip() == INSTRUCTION_TEMPLATE.strip()
    assert memory.k == memory_key
    assert memory.return_messages is True
    assert memory.input_key == "question"
    assert memory.memory_key == "history"

    # Test case 2
    model_type = "gpt3"
    persona = "custom"
    memory_key = 5

    prompt, memory = stepback_prompt(model_type, persona, memory_key)

    # Assert the prompt and memory objects are returned correctly
    assert isinstance(prompt, PromptTemplate)
    assert isinstance(memory, ConversationBufferWindowMemory)
    assert prompt.input_variables == ["normal_context", "step_back_context", "question"]
    assert prompt.template.strip() == INSTRUCTION_TEMPLATE.strip()
    assert memory.k == memory_key
    assert memory.return_messages is True
    assert memory.input_key == "question"
    assert memory.memory_key == "history"