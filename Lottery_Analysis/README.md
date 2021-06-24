![lotto_649_-1](https://user-images.githubusercontent.com/70767722/123316133-83053080-d4fa-11eb-9c25-a961875c75d1.jpeg)


# 6/49 Lottery Analysis 


I'm personally not a big lottery player. However, as everyone else, I like the idea of winning the lottery and having such a lot of money in just a night (if you are the winner). Some people play lottery for fun, but soon it becomes a habit which eventually turns into an addiction. Besides that the lottery company's marketing team is doing such a great job creating a belief of everyone can be a millionaire [Lottery Winners](https://www.playnow.com/lottery/winners/you-could-be-next/). In Ontario province, we have OLG which is the Ontario government agency that delivers lottery and gaming entertainment. They provide variety lottery games; for instance, [LOTTO MAX](https://www.olg.ca/en/lottery/play-lotto-max-encore.html) that reaches `$70 million` on Tuesday, June 22, 2021 and [LOTTO 6/49](https://www.olg.ca/en/lottery/play-lotto-649-encore/about.html) that always guarantes `$1 million` in every draw night and maximum with `$5 millio`n on Wednesday, June 22, 2021. 

In this analysis, we just focus on the **6/49** lottery, create some functions and find the insights from the data set [649](https://www.kaggle.com/datascienceai/lottery-dataset) on Kaggle that has 3,665 drawings from 1982 to 2018.

Through the analysis, we want to find the answers for the following questions:
* What is the probability of winning the big prize with a single ticket?
* What is the probability of winning the big prize if we play 50 different tickets (or any different number)?
* What is the probability of having at least five (or four, or three, or two) winning numbers on a single ticket?

# 6/49 Gameplay


As the name implies, six numbers are drawn from a set of 49. if a ticke matches all six numbers, the jackpot prize of at least `$5 million` is won. There is a bonus number is drawn and if a player's ticket matches five numbers plus the bonus number then the player wins the "second prize" which is usually between `$100,000 and $500,000`.

In case there are more than one player win the top or second prize, it will be split amongst them. Lesser prizes are also awarded if one matches at least two numbers. If the top prize is not won, the jackpot prize increases for the next draw.

# Creating Functions


From the 6/49 gameplay, we know how the game works. For example, if a player bought a ticket with the numbers {07,02,19,91,11,56}, he/she only wins the big prize if the six numbers on the tickets match all the drawn six numbers as {07,02,19,91,11,56} (the order does not matter).

* We will create a function that can calculate the probability of winning the big prize with the various numbers they play on a single ticket.

* Then we can compare their ticket with the historical lottery data and consider whether they might win or not. Since the data set consists of 3,665 drawings dating from 1982 to 2018. For each drawing, we can find the six numbers drawn in the following 6 columns:

    - NUMBER DRAWN 1
    - NUMBER DRAWN 2
    - NUMBER DRAWN 3
    - NUMBER DRAWN 4
    - NUMBER DRAWN 5
    - NUMBER DRAWN 6

* We want to create another function to check the number of times the combination selected occured in the 6/49 data set as well as the probability of winning the big prize in the next drawing with that combination.

* Besides the big prize, we can also win the smaller prizes if the players' ticket match at least two, three, four, or five of the six numbers drawn. Therefore, the players might be interested in knowing the probability of having these two, three, four, or five winning numbers.

    * In the function, the user can input:
        * six different numbers from 1 to 49
        * an interger between 2 and 5 that represents the number of winning numbers expected.
    * The function will print information about the probabity of having the input number of winning numbers.
    * The function will not return the probability of having at least five winning numbers.

# Summary

In this analysis, we will be creating four functions that can help us to understand the odds of winning the 6/49 lottery from the past data:


|    <strong>Functions</strong>        |  Description                                        |
|-------------------|-----------------------------------------------------|
|<strong>big_prize_proba() </strong>  | Calculates the probability of winnig the big prize with a one ticket    |   
|<strong>check_past_winning()</strong>         | Checks whether a certain combination has occurred in 6/49 data set     |   
|<strong>check_multi_proba()</strong>         | Calculates the probability for any number of the ticket  |
|<strong>check_less_6_proba()</strong>   |  Calculates the probability of having two, three, four, or five winning numbers  |
