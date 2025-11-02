# AI Hackathon: Code. Create. Conquer.

## Stream 1 (advanced): Automated Resume Matching System for Job Descriptions

### Goal

Build a system that automatically matches candidate resumes with open job positions using:

* text analysis
* semantic similarity
* key characteristics of both the candidate and the employer

The system must output:

1. a score showing how well a candidate fits a given position
2. an explanation of the evaluation

### Problem Description

Modern HR departments receive a large number of resumes and have many open positions, which makes manual selection inefficient and time-consuming. Automating this process helps:

* identify the most suitable candidates faster
* reduce the workload and save resources

The task: given a job description and a resume, the system should rank candidates for that job.

### Input Data

#### 1. Resume (PDF) structure

* Full name
* Desired position
* Preferred work format
* Willingness to travel
* Contact details
* Work authorization
* Work experience

  * Professional experience (job title, company, responsibilities)
* Education level
* Professional skills and competencies
* Additional information

#### 2. Job description (PDF) structure

* Company name and brief description
* Job title
* Location
* Candidate requirements (work experience, skills, education, work authorization)
* Responsibilities and employer expectations

---

## Evaluation Criteria

Solutions will be assessed on four areas:

### 1. Model Quality (50%)

**Evaluation data:**
A closed dataset of job descriptions and resumes will be used, where the correct ranking order is already known.

**Procedure:**

1. Each “job–resume” pair is passed through the team’s automatic scoring system.
2. The system outputs scores.
3. Scores are used to build a ranking of resumes for each job.
4. Relative ranking quality matters more than absolute score values.

**Metric:**

* **NDCG (Normalized Discounted Cumulative Gain)** — measures how close the produced ranking is to the ideal ranking.

### 2. Code Quality (20%)

Assessed by:

* compliance with linters (style and code quality analyzers)
* clarity of project architecture
* ease of maintenance and scalability

### 3. Performance (10%)

Assessed by:

* program response time
* efficiency and cost of third-party service usage
  Lower latency and cheaper external calls = higher score.

### 4. Presentation Quality (20%)

Teams must present their solution to the jury. Assessed by:

* clarity of the idea and technical explanation
* effectiveness of presentation and product demo
* ability to answer jury questions clearly and convincingly

---

## Technical Requirements

* **Programming language:** Python
* **API:** ability to integrate external NLP or LLM models (team’s choice)
* **Frameworks and libraries:** open choice