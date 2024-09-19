# Contributing

We welcome contribution from the community.
The project is - as is CleanRL - under MIT license which is a very permissive license. 


## Getting Started with Contributions
To contribute to this project, please follow these steps:

### 1. Clone and Fork the Repository
First, clone the repository using the following command:
```bash
git clone https://github.com/pytorch-labs/leanrl.git
```

Then, fork the repository by clicking the "Fork" button on the top-right corner of the GitHub page. 
This will create a copy of the repository in your own account.
Add the fork to your local list of remote forks:
```bash
git remote add <my-github-username> https://github.com/<my-github-username>/leanrl.git
```

### 2. Create a New Branch
Create a new branch for your changes using the following command:
```bash
git checkout -b [branch-name]
```
Choose a descriptive name for your branch that indicates the type of change you're making (e.g., `fix-bug-123`, `add-feature-xyz`, etc.).
### 3. Make Changes and Commit
Make your changes to the codebase, then add them to the staging area using:
```bash
git add <files-to-add>
```

Commit your changes with a clear and concise commit message:
```bash
git commit -m "[commit-message]"
```
Follow standard commit message guidelines, such as starting with a verb (e.g., "Fix", "Add", "Update") and keeping it short.

### 4. Push Your Changes
Push your changes to your forked repository using:
```bash
git push --set-upstream <my-github-username> <branch-name>
```

### 5. Create a Pull Request
Finally, create a pull request to merge your changes into the main repository. Go to your forked repository on GitHub, click on the "New pull request" button, and select the branch you just pushed. Fill out the pull request form with a clear description of your changes and submit it.

We'll review your pull request and provide feedback or merge it into the main repository.
