from setuptools import setup, find_packages

setup(
    name='agent-cli',
    version='1.0.0',
    description='AI agent with smart command classifier for safe system operations',
    author='Your Name',
    author_email='your@email.com',
    url='https://github.com/yourname/agent-cli',
    packages=find_packages(),
    install_requires=[
        'ollama>=0.3.0',
        'langgraph>=0.0.1',
        'langchain>=0.1.0',
        'typing-extensions>=4.0.0',
    ],
    entry_points={
        'console_scripts': [
            'agent-cli=agent_cli.main:main',
        ],
    },
    python_requires='>=3.9',
)