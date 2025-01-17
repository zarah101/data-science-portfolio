import matplotlib.pyplot as plt
import seaborn as sns
import time
import random

# Function to display a fun and engaging greeting
def portfolio_greeting(name="Visitor"):
    greeting_message = f"""
    🎉🎉 Hey {name}! Welcome to my Data Science Portfolio! 🎉🎉
    
    I'm a data enthusiast who lives and breathes numbers, patterns, and insights!
    I specialize in:
    🔍 Data Analysis
    🤖 Machine Learning
    📊 Data Visualization
    💻 Python, PySpark, Pandas
    And much more...

    I believe in the power of data to tell compelling stories and solve real-world problems. 
    Let's explore the world of data science together! 📈✨
    """
    
    print(greeting_message)
    
    # Pausing for dramatic effect (lol)
    time.sleep(2)

    print("Now, let me show you a fun visualization of my skills...")

# Display the creative greeting
portfolio_greeting("You")

# Create a fun and colorful visual with Seaborn
categories = ['Data Science', 'Machine Learning', 'Python', 'Visualization', 'Storytelling']
values = [random.randint(70, 95) for _ in categories]  # Randomizing to add a fun touch

# Set the color palette for the chart
sns.set_palette("Set2")

fig, ax = plt.subplots(figsize=(8, 5))

# Creating a horizontal bar chart
ax.barh(categories, values, color=sns.color_palette("Set2"))

# Add a title and customize it
ax.set_title("My Skills Breakdown 🌟", fontsize=18, color='darkblue', weight='bold')
ax.set_xlabel('Proficiency (%)', fontsize=12, color='black')
ax.set_ylabel('Skills', fontsize=12, color='black')

# Customize the tick labels for fun and style
ax.tick_params(axis='y', labelsize=12, labelcolor='black')

# Show the plot with cool visuals!
plt.tight_layout()
plt.show()


