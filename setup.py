import setuptools

if __name__ == "__main__":

    with open('requirements.txt', 'r') as f:
        requirements = f.readlines()
        requirements = [line.strip() for line in requirements if line.strip()]

    setuptools.setup(name = 'pipelines',
    version = '0.1.0',
    author = 'Dillon Wong',
    author_email = '',
    description = 'Build data pipelines',
    url = 'https://github.com/dilwong/pipelines',
    install_requires = requirements,
    packages=['pipelines'],
    package_dir={'pipelines': 'src'}
    )