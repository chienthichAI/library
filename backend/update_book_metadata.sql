-- SQL Script to enrich library book catalog for RAG Chatbot
-- Date: 2025-05-22

-- 1. Programming & IT Masterpieces
UPDATE books SET 
    description = 'Cuốn sách của "Uncle Bob" Robert C. Martin giải thích cách xây dựng kiến trúc phần mềm sạch, tách biệt logic nghiệp vụ khỏi các chi tiết kỹ thuật. Nó nhấn mạnh các nguyên tắc SOLID, Dependency Rule và tầm quan trọng của việc duy trì sự chính trực trong thiết kế hệ thống để đảm bảo khả năng bảo trì lâu dài.',
    subject_category = 'Software Architecture',
    publisher = 'Pearson / Prentice Hall',
    publication_year = 2017,
    isbn_13 = '9780134494166'
WHERE title LIKE '%Clean Architecture%';

UPDATE books SET 
    description = 'Tác phẩm kinh điển về lập trình, tập trung vào cách viết mã nguồn dễ đọc, dễ hiểu và dễ bảo trì. Sách giới thiệu các tiêu chuẩn đặt tên, cấu trúc hàm, xử lý lỗi và các kỹ thuật "refactoring" nhỏ để biến mã nguồn từ "tạm ổn" thành "tốt".',
    subject_category = 'Programming Best Practices',
    publisher = 'Pearson / Prentice Hall',
    publication_year = 2008,
    isbn_13 = '9780132350884'
WHERE title LIKE '%Clean Code%';

UPDATE books SET 
    description = 'Cuốn sách "vỡ lòng" về 23 mẫu thiết kế phần mềm kinh điển của nhóm "Gang of Four". Nó cung cấp các giải pháp chuẩn cho những vấn đề thiết kế phổ biến trong lập trình hướng đối tượng, giúp tạo ra các hệ thống linh hoạt, thanh thoát và dễ tái sử dụng.',
    subject_category = 'Design Patterns',
    publisher = 'Addison-Wesley',
    publication_year = 1994,
    isbn_13 = '9780201633610'
WHERE title LIKE '%Design Patterns%' AND author LIKE '%Gamma%';

UPDATE books SET 
    description = 'Hướng dẫn toàn diện về Docker từ cơ bản đến nâng cao. Nigel Poulton giúp người đọc nắm vững khái niệm container, images, Docker Engine, cùng các công cụ như Docker Compose và Swarm để triển khai và quản lý ứng dụng cloud-native hiệu quả.',
    subject_category = 'DevOps & Containers',
    publisher = 'Independently Published',
    publication_year = 2020,
    isbn_13 = '9781521822807'
WHERE title LIKE '%Docker Deep Dive%';

UPDATE books SET 
    description = 'Cuốn sách gối đầu giường của mọi lập trình viên Java. Joshua Bloch chia sẻ 90 quy tắc vàng để sử dụng ngôn ngữ Java một cách hiệu quả nhất, bao gồm cả các tính năng mới trong Java 8 và 9 như Lambda và Streams.',
    subject_category = 'Java Programming',
    publisher = 'Addison-Wesley',
    publication_year = 2017,
    isbn_13 = '9780134685991'
WHERE title LIKE '%Effective Java%';

UPDATE books SET 
    description = 'Một hành trình khám phá thế giới lập trình thông qua JavaScript. Sách không chỉ dạy cú pháp mà còn đi sâu vào cấu trúc dữ liệu, thuật toán, lập trình hướng đối tượng và cách tương tác với trình duyệt cũng như Node.js.',
    subject_category = 'JavaScript / Web Development',
    publisher = 'No Starch Press',
    publication_year = 2018,
    isbn_13 = '9781593279509'
WHERE title LIKE '%Eloquent JavaScript%';

UPDATE books SET 
    description = 'Cuốn sách giúp bạn hiểu sâu về bản chất của Python để viết mã nguồn đúng chất "Pythonic". Nó bao gồm các chủ đề nâng cao như mô hình dữ liệu, hàm hạng nhất, coroutines và siêu lập trình (metaprogramming).',
    subject_category = 'Python Programming',
    publisher = 'O''Reilly Media',
    publication_year = 2022,
    isbn_13 = '9781492056355'
WHERE title LIKE '%Fluent Python%';

UPDATE books SET 
    description = 'Bộ "Kinh thánh" về thuật toán được sử dụng rộng rãi trong các trường đại học toàn cầu. Sách trình bày chi tiết về thiết kế và phân tích thuật toán từ cơ bản như sắp xếp đến nâng cao như đồ thị và quy hoạch động bằng mã giả chuyên nghiệp.',
    subject_category = 'Algorithms / Computer Science',
    publisher = 'MIT Press',
    publication_year = 2022,
    isbn_13 = '9780262046305'
WHERE title LIKE '%Introduction To Algorithms%';

UPDATE books SET 
    description = 'Dẫn dắt các lập trình viên Python từ trình độ cơ bản đến chuyên nghiệp. Sách tập trung vào việc tổ chức dự án lớn, trừu tượng hóa mã nguồn đúng mức và các nguyên tắc thiết kế phần mềm để mã nguồn dễ bảo trì.',
    subject_category = 'Python Programming',
    publisher = 'Manning Publications',
    publication_year = 2020,
    isbn_13 = '9781617296086'
WHERE title LIKE '%Practices of the Python Pro%';

UPDATE books SET 
    description = 'Cuốn sách dự án thực tế dành cho người mới bắt đầu học Python. Người đọc sẽ được thực hành qua 3 dự án: xây dựng trò chơi video, tạo trực quan hóa dữ liệu và phát triển ứng dụng web tương tác.',
    subject_category = 'Python Programming / Beginners',
    publisher = 'No Starch Press',
    publication_year = 2015,
    isbn_13 = '9781593276034'
WHERE title LIKE '%Python Crash Course%';

UPDATE books SET 
    description = 'Kỹ thuật cải thiện thiết kế mã nguồn hiện có mà không làm thay đổi hành vi bên ngoài. Martin Fowler cung cấp danh mục hơn 70 kỹ thuật refactoring kèm theo các "mùi mã" (code smells) để lập trình viên nhận biết khi nào cần làm sạch mã.',
    subject_category = 'Software Engineering',
    publisher = 'Addison-Wesley',
    publication_year = 2018,
    isbn_13 = '9780134757599'
WHERE title LIKE '%Refactoring%';

UPDATE books SET 
    description = 'Cuốn sách căn bản về ngôn ngữ lập trình C, được viết bởi chính cha đẻ của nó. Đây là tài liệu tham khảo chuẩn mực về cú pháp, kiểu dữ liệu và các tính năng cốt lõi của C, giúp hình thành tư duy lập trình hệ thống.',
    subject_category = 'C Programming',
    publisher = 'Prentice Hall',
    publication_year = 1988,
    isbn_13 = '9780131103627'
WHERE title LIKE '%The C Programming Language%';

UPDATE books SET 
    description = 'Tổng hợp các lời khuyên thực tế về kỹ thuật phần mềm để giúp lập trình viên nâng cao kỹ năng, quản lý sự nghiệp và viết mã nguồn linh hoạt, dễ thích nghi. Phiên bản kỷ niệm 20 năm cập nhật các xu hướng hiện đại.',
    subject_category = 'Software Development',
    publisher = 'Addison-Wesley',
    publication_year = 2019,
    isbn_13 = '9780135957059'
WHERE title LIKE '%The Pragmatic Programmer%';

UPDATE books SET 
    description = 'Môt chuỗi các cuốn sách đào sâu vào những góc khuất và khái niệm dễ gây nhầm lẫn nhất của JavaScript như types, closure, this, và prototypes. Kyle Simpson khuyến khích lập trình viên thực sự hiểu bản chất ngôn ngữ thay vì chỉ dùng trên bề mặt.',
    subject_category = 'JavaScript Programming',
    publisher = 'O''Reilly Media',
    publication_year = 2015,
    isbn_13 = '9781491924464'
WHERE title LIKE '%You Don''t Know JS%';

-- 2. Vietnamese Culture & Skills
UPDATE books SET 
    description = 'Cẩm nang về các quy tắc ứng xử, truyền thống và giá trị văn hóa đặc trưng của Việt Nam. Giúp người đọc hiểu sâu hơn về lịch sử, thay đổi xã hội và cách giao tiếp hiệu quả, tránh những hiểu lầm không đáng có trong môi trường địa phương.',
    subject_category = 'Văn hóa & Du lịch',
    publisher = 'Kuperard',
    publication_year = 2006,
    isbn_13 = '9781787028524'
WHERE title LIKE '%Culture Smart%';

UPDATE books SET 
    description = 'Cuốn sách "vỡ lòng" về 5000 từ vựng tiếng Nhật thông dụng nhất, được phân loại theo chủ đề để người học dễ dàng tiếp cận và ghi nhớ. Thích hợp cho sinh viên và người đi làm muốn củng cố vốn từ giao tiếp nhanh chóng.',
    subject_category = 'Học Ngoại Ngữ / Tiếng Nhật',
    publisher = 'NXB Hồng Đức',
    publication_year = 2019,
    isbn_13 = '8935246926918'
WHERE title LIKE '%5000 từ vựng tiếng Nhật%';

UPDATE books SET 
    description = 'Sách song ngữ Việt - Trung mang tính chất truyền cảm hứng tích cực. Tập hợp 101 câu nói ý nghĩa về cuộc sống, sự trưởng thành và yêu bản thân, giúp người đọc vừa học tiếng Trung vừa bồi đắp tâm hồn.',
    subject_category = 'Phát triển bản thân / Song ngữ',
    publisher = 'NXB Thanh Niên',
    publication_year = 2023,
    isbn_13 = '9786043228175'
WHERE title LIKE '%Dẫu bình thường%';

UPDATE books SET 
    description = 'Tài liệu cung cấp kiến thức pháp luật toàn diện phục vụ kinh doanh và quản lý tại Việt Nam. Nội dung bao gồm Luật kinh tế, Luật doanh nghiệp, Luật lao động và các quy định về tài chính, môi trường.',
    subject_category = 'Pháp luật / Kinh tế',
    publisher = 'NXB Tư pháp',
    publication_year = 2023,
    isbn_13 = '9786048126261'
WHERE title LIKE '%Luật Kinh tế%';

UPDATE books SET 
    description = 'Cuốn sách thách thức các quan điểm marketing truyền thống, nhấn mạnh rằng marketing thực thụ phải mang lại kết quả tài chính cụ thể. Tác giả Sergio Zyman chia sẻ kinh nghiệm từ Coca-Cola về cách tạo ra lợi nhuận bền vững.',
    subject_category = 'Marketing / Kinh doanh',
    publisher = 'NXB Văn Hóa – Văn Nghệ',
    publication_year = 2020
WHERE title LIKE '%Marketing giỏi phải kiếm được tiền%';

UPDATE books SET 
    description = 'Phân tích các hình thức "khủng bố" trên mạng xã hội và hướng dẫn quy trình xử lý khủng hoảng truyền thông chuyên nghiệp. Giúp doanh nghiệp bảo vệ hình ảnh và danh tiếng trước những luồng dư luận tiêu cực.',
    subject_category = 'Truyền thông / Quản trị',
    publisher = 'NXB Tổng Hợp TPHCM',
    publication_year = 2018
WHERE title LIKE '%Quản trị "khủng bố" trực tuyến%';

UPDATE books SET 
    description = 'Cẩm nang hướng dẫn cách sử dụng các kênh mạng xã hội hiệu quả để truyền dữ liệu và kết nối thương hiệu. Sách cung cấp các mẹo tối ưu hóa chiến lược truyền thông và tạo sự tin tưởng từ khách hàng.',
    subject_category = 'Kỹ năng / Truyền thông',
    publisher = 'NXB Phụ Nữ Việt Nam',
    publication_year = 2021,
    isbn_13 = '9786043297218'
WHERE title LIKE '%Truyền sao cho thông%';

UPDATE books SET 
    description = 'Tập hợp các bài tiểu luận sâu sắc về sự hình thành bản sắc văn hóa Việt Nam qua lịch sử. Hữu Ngọc bàn về cách truyền thống luôn thay đổi và thích nghi thông qua các cuộc giao thoa văn hóa đầy biến động.',
    subject_category = 'Văn hóa / Lịch sử',
    publisher = 'NXB Thế Giới',
    publication_year = 2018,
    isbn_13 = '9786047745067'
WHERE title LIKE '%Viet Nam: Tradition and Change%';

-- 3. Language & Coursebooks
UPDATE books SET 
    description = 'Tài liệu chuẩn bị cho kỳ thi IELTS Academic với các đề thi thực tế từ những năm trước. Bao gồm đầy đủ 4 kỹ năng Listening, Reading, Writing và Speaking với đáp án và phân tích chi tiết.',
    subject_category = 'Luyện thi IELTS',
    publisher = 'Cambridge University Press',
    publication_year = 2021
WHERE title LIKE '%Cambridge IELTS Academic 16%';

UPDATE books SET 
    description = 'Tài liệu luyện thi IELTS cập nhật với các đề bài mới nhất, giúp thí sinh làm quen với cấu trúc bài thi và tiêu chí chấm điểm để đạt mục tiêu band điểm cao cho mục đích học thuật.',
    subject_category = 'Luyện thi IELTS',
    publisher = 'Cambridge University Press',
    publication_year = 2022
WHERE title LIKE '%Cambridge IELTS Academic 17%';

UPDATE books SET 
    description = 'Cuốn sách mới nhất trong series Cambridge IELTS, cung cấp các bài test sát thực tế nhất cho kỳ thi năm 2024. Đi kèm với file âm thanh và các bài mẫu viết có nhận xét của giám khảo.',
    subject_category = 'Luyện thi IELTS',
    publisher = 'Cambridge University Press',
    publication_year = 2024
WHERE title LIKE '%Cambridge IELTS Academic 18%';

UPDATE books SET 
    description = 'Giáo trình tiếng Anh thương mại cấp độ B2 của Pearson phối hợp với Financial Times. Giúp người học phát triển các kỹ năng giao tiếp chuyên nghiệp trong môi trường công sở quốc tế.',
    subject_category = 'Tiếng Anh chuyên ngành',
    publisher = 'Pearson',
    publication_year = 2021,
    isbn_13 = '9781292248585'
WHERE title LIKE '%Business Partner B2%';

UPDATE books SET 
    description = 'Giáo trình tiếng Nhật sơ cấp tập trung vào khả năng ứng dụng thực tế. Giúp người học làm quen với các tình huống giao tiếp đời thường và bồi dưỡng cả 4 kỹ năng nghe, nói, đọc, viết.',
    subject_category = 'Học Ngoại Ngữ / Tiếng Nhật',
    publisher = 'NXB ALC Press',
    publication_year = 2011,
    isbn_13 = '9784757419773'
WHERE title LIKE '%Dekiru Nihongo%' OR title LIKE '%Tiếng Nhật trong tầm tay%';

UPDATE books SET 
    description = 'Hành trình khám phá 29 quốc gia Châu Á qua 45 bài đọc tiếng Anh sinh động. Vừa giúp nâng cao vốn từ vựng, vừa cung cấp kiến thức văn hóa về các địa danh nổi tiếng như Angkor Wat, Vạn Lý Trường Thành.',
    subject_category = 'Học Ngoại Ngữ / Du lịch',
    publisher = 'ZenBooks',
    publication_year = 2023,
    isbn_13 = '8794069304569'
WHERE title LIKE '%Pack your bags%';

-- 4. General & Others
UPDATE books SET 
    description = 'Cuốn sách tô màu chủ đề các loài chim, giúp trẻ em khám phá thiên nhiên và phát triển kỹ năng hội họa cơ bản thông qua màu sắc sinh động.',
    subject_category = 'Thiếu nhi / Giải trí',
    publisher = 'NXB Mỹ Thuật',
    publication_year = 2022
WHERE title LIKE '%BÉ TÔ MÀU%';

UPDATE books SET 
    description = 'Tài liệu hướng dẫn thiết kế các yếu tố đồ họa cơ bản như bố cục, màu sắc và typography. Là tài liệu tham khảo hữu ích cho các nhà thiết kế trong việc truyền tải thông điệp thị giác hiệu quả.',
    subject_category = 'Đồ họa / Thiết kế',
    publisher = 'Rockport Publishers',
    publication_year = 2020,
    isbn_13 = '9781631598722'
WHERE title LIKE '%Design Elements%';

UPDATE books SET 
    description = 'Vở thực hành và ghi chép hỗ trợ quá trình học tập cá nhân, giúp tổ chức kiến thức khoa học và hiệu quả.',
    subject_category = 'Dụng cụ học tập',
    publisher = 'Chiendz / Local Pub',
    publication_year = 2023
WHERE title LIKE '%Vở học tập%';
