

'''
Of the ID classes (which form a single high-level partition), we need 3 subsets:
    ID1) ID classes
        a) Training ID
        b) Validation ID
        c) Test ID

Of the OOD classes, we have 3 high-level partitions:
    OOD1) Training OOD Classes (OE data)
    OOD2) Validation OOD Classes
    OOD3) Test OOD Classes

Question: Should training OOD classes show up in validation set or test set?

Answer:   No. I don't think they should. At each phase, we want to present the network with previously unseen anomaly types.
          Because of this, we can create 3 monolithic datasets from OOD classes.

To brainstorm a bit... 
    1) We could take CIFAR-100 and split it into its train and test parts
    2) For both parts, build all four partitions. You should have 8 partitions in total.
    3) Two of these partitions will be ID partitions: Raw Training ID and Test ID
        a) Partition Raw Training ID into an 80/20 split. Call the 80% partition the Training ID partition and
           the 20% partition the Validation ID partition. At this point, you will have 9 total partitions; three of 
           which will be for ID data.
    4) You still have 6 OOD partitions... what do we do with these? Throw out the Training OOD class "test" set. This will create a more 
       balanced training process. Concatenate the sets corresponding to validation OOD classes and to test OOD classes. 
       You should now have 6 partitions.
    5) 

ID Class dataset should be handled by building a dataset according to these indices. This will be a 10 class subset of 
CIFAR-100. From there, we can
'''