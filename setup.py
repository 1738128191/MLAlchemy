from setuptools import setup, find_packages

setup(
    name="my",  # 包名
    version="1.0.1",  # 当前版本号，后续更新时修改
    author="ZiFan Xv",
    author_email="1738128191@qq.com",
    description="Simplifying Advanced API Access for Data Mining & Machine Learning",
    url="https://github.com/1738128191/MLAlchemy",
    packages=find_packages(),  # 包含的子包列表
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires='>=3.9',  # 支持的Python版本
)

