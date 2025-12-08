pipeline {
    agent any

    environment {
        IMAGE_NAME = 'asif1993/second_ml'
    }

    stages {
        
        stage('Checkout') {
            steps { git branch: 'main', credentialsId: 'git-cred', url: 'https://github.com/AsifShaikh93/first_ml.git' }
        }

        stage('Build & Push Docker Image') {
            steps {
                script {
                    def shortCommit = sh(script: "git rev-parse --short HEAD",returnStdout: true).trim()
                    env.IMAGE_TAG = shortCommit
                }

                withCredentials([usernamePassword(credentialsId: 'docker_cred', usernameVariable: 'user', passwordVariable: 'pass')]) {
                    sh """
                      echo "$pass" | docker login -u "$user" --password-stdin

                      docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
                      docker push ${IMAGE_NAME}:${IMAGE_TAG}

                      docker logout
                    """
                }
            }
        }
    }
    
        post {
          success {
            script {
               withCredentials([usernamePassword(credentialsId: 'git-cred',usernameVariable: 'user',passwordVariable: 'pass')]) {
                sh """
                  sed -i "s#^\\( *image: \\).*#\\1${IMAGE_NAME}:${IMAGE_TAG}#" k8s/deploy.yaml

                  git config user.name "Jenkins CI"
                  git config user.email "jenkins@local"
                  git remote set-url origin https://${user}:${pass}@github.com/AsifShaikh93/first_ml.git

                  git add k8s/deploy.yaml
                  if git diff --cached --quiet; then
                    echo "No changes to commit"
                  else
                    git commit -m "Update image tag to ${IMAGE_TAG} [ci skip]"
                    git push origin main
                  fi
                """
            }
        }
    }
}


