#
# DUMP FILE
#
# Database is ported from MS Access
#------------------------------------------------------------------
# Created using "MS Access to MySQL" form http://www.bullzip.com
# Program Version 5.4.274
#
# OPTIONS:
#   sourcefilename=C:\Data\Book - exploring IT\exploring IT CAPS Series\Grade 11\CDCollection.accdb
#   sourceusername=
#   sourcepassword=
#   sourcesystemdatabase=
#   destinationdatabase=CDCollection
#   storageengine=InnoDB
#   dropdatabase=0
#   createtables=1
#   unicode=1
#   autocommit=1
#   transferdefaultvalues=1
#   transferindexes=1
#   transferautonumbers=1
#   transferrecords=1
#   columnlist=1
#   tableprefix=
#   negativeboolean=0
#   ignorelargeblobs=0
#   memotype=LONGTEXT
#   datetimetype=DATETIME
#

CREATE DATABASE IF NOT EXISTS `CDCollection`;
USE `CDCollection`;

#
# Table structure for table 'tblCD'
#

DROP TABLE IF EXISTS `tblCD`;

CREATE TABLE `tblCD` (
  `ID` INTEGER AUTO_INCREMENT, 
  `Title` VARCHAR(30), 
  `Artist` VARCHAR(20), 
  `NoOfSongs` TINYINT(3) UNSIGNED DEFAULT 0, 
  `Genre` VARCHAR(20), 
  INDEX (`ID`)
) ENGINE=innodb DEFAULT CHARSET=utf8;

SET autocommit=1;

#
# Dumping data for table 'tblCD'
#

INSERT INTO `tblCD` (`ID`, `Title`, `Artist`, `NoOfSongs`, `Genre`) VALUES (1, 'Greatest Hits', 'Lighthouse Family', 19, 'R&B');
INSERT INTO `tblCD` (`ID`, `Title`, `Artist`, `NoOfSongs`, `Genre`) VALUES (2, 'Crime of the Century', 'Supertramp', 8, 'Rock');
INSERT INTO `tblCD` (`ID`, `Title`, `Artist`, `NoOfSongs`, `Genre`) VALUES (3, 'No Angel', 'Dido', 12, 'R&B');
INSERT INTO `tblCD` (`ID`, `Title`, `Artist`, `NoOfSongs`, `Genre`) VALUES (4, 'Zooropa', 'U2', 14, 'Rock');
INSERT INTO `tblCD` (`ID`, `Title`, `Artist`, `NoOfSongs`, `Genre`) VALUES (5, 'Jagged Little Pill', 'Alanis Morrisette', 12, 'Rock');
INSERT INTO `tblCD` (`ID`, `Title`, `Artist`, `NoOfSongs`, `Genre`) VALUES (6, 'Our Town', 'Deacon Blue', 19, 'Pop');
INSERT INTO `tblCD` (`ID`, `Title`, `Artist`, `NoOfSongs`, `Genre`) VALUES (7, 'Greatest Hits', 'Deacon Blue', 19, 'Pop');
INSERT INTO `tblCD` (`ID`, `Title`, `Artist`, `NoOfSongs`, `Genre`) VALUES (8, 'Joshua Tree', 'U2', 11, 'Pop');
INSERT INTO `tblCD` (`ID`, `Title`, `Artist`, `NoOfSongs`, `Genre`) VALUES (11, 'Gold Greatest Hits', 'Abba', 18, 'Pop');
INSERT INTO `tblCD` (`ID`, `Title`, `Artist`, `NoOfSongs`, `Genre`) VALUES (12, 'Gold Greatest Hits', 'Abba', 18, 'Pop');
# 10 records

