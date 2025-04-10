# 🧠 Machine Learning Engineer Take-Home Test

## ⏱️ Overview

This take-home assessment is part of our talent screening process. It is designed to evaluate your ability to reason through a forecasting problem and communicate your approach effectively.

**Time Commitment:** 2–4 hours  

## 📦 Forecast Weekly Box Demand by Box Type

### 📘 Background

At Oddbox, we sell **8 different types of subscription boxes** on a weekly basis. Order volumes fluctuate depending on:
- Seasonality
- Customer behavior
- Marketing efforts

Customers may have either a **weekly** or **fortnightly** subscription.

## 🎯 Your Task

You’ll receive **12 months of historical data** on weekly box orders by type.

### ✅ Objectives

1. **Clone Repo**  
   Clone this repo under your own Github (or other git provider) account.

2. **Forecast Total Box Orders**  
   Predict the number of box orders for each box type for the **next 4 weeks**.

3. **Explain Your Reasoning**
   - Why did you choose this forecasting method?
   - What assumptions are you making?
   - What would you do differently with more time or data?
   - How would you incorporate future known events (e.g., holidays, promotions)?
   - How would you measure forecast performance in production?

## 🚀 Stretch Goals

These are optional, but we'd love to hear your thoughts—either implemented in code or prepared for discussion during our call:

> 🥕 **Box Customisation Challenge**  
At Oddbox, customers can **swap produce items** in their box with one of **N alternatives**.  
How would you approach incorporating this customisation logic into your demand forecast?

## 📊 Data

In this repository you'll find a file called `data.csv`. This is your input data for the forecast model. Please feel free to generate additional data as needed, or consider what other data inputs you would need to build an effective model.

| Column Name             | Description                                                       |
|-------------------------|-------------------------------------------------------------------|
| `Week`                  | The delivery week                                                 |
| `Box Type`              | The type of box (one of 8 types)                                  |
| `Box Orders`            | Number of box orders delivered that week                          |
| `Is Marketing Week`     | Whether any promotions or marketing campaigns were running        |
| `Holiday Week`          | Whether the week falls within a holiday period                    |
| `Weekly Subscribers`    | Number of customers with a weekly subscription                    |
| `Fortnightly Subscribers` | Number of customers with a fortnightly subscription             |

## 📦 Submission Instructions

- Include a **brief write-up** of your approach (Markdown or Jupyter Notebook is fine).
- You may use **any tools or languages** you prefer.
- Make your code readable and well-documented.
- Share a link to the repository with the hiring manager for the role.
